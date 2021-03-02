try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from generate_map import generateXZ, getXML
import survivaiVISION
from survivaiVISION import draw_helper

import torch
from torch import nn
import torch.nn.functional as F


colors = {'wood': (162, 0, 93)}

class PixelViewModel(TorchModelV2, nn.Module):
    # default functions from 
    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1) # 32, 432, 240
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32, 432, 240
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32, 432, 240
        self.policy_layer = nn.Linear(32 * 432 * 240, 6)
        self.value_layer = nn.Linear(32 * 432 * 240, 1)
        self.value = None
    
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs'].float()  # BATCH size 4, 432, 240

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.flatten(start_dim=1)

        policy = self.policy_layer(x)
        self.value = self.value_layer(x)
        
        return policy, state

    def value_function(self):
        return self.value.squeeze(1)

class SurvivAI(gym.Env):
    def __init__(self, env_config):
        # RLLib parameters
        self.episode_return = 0
        self.episode_step = 0
        self.returns = []
        self.steps = []
        self.obs = None
        
        # static variables
        self.log_frequency = 1
        self.obs_size = 5
        self.action_space = Box(low=np.array([-0.5, -0.25, -0.5]), high=np.array([1.0, 0.5, 1.0]), dtype=np.float32)
        self.observation_space = Box(0, 255, shape=(4,432,240), dtype=np.int32)
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'attack 1'  # Destroy block
        }

        # Malmo parameters
        self.agent_host = MalmoPython.AgentHost()

        #Set video policy and create drawer
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
        self.canvas = survivaiVISION.canvas
        self.root = survivaiVISION.root
        self.drawer = draw_helper(self.canvas)

        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:',e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)

        self.can_break = False #do we need this?
    
    def step(self, action):
        """
        Take an action in the environment and return the results.
        Args
            action: <int> index of the action to take
        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        print(action)

        # Assign amounts to move, turn, and attack by
        world_state = self.agent_host.getWorldState()
        move_command, turn_command, attack_command = self.checkForWood(world_state, action)

        print("{}, {}, {}".format(move_command,turn_command,attack_command))

        #do appropriate move and turn
        self.agent_host.sendCommand(move_command)
        self.agent_host.sendCommand(turn_command)

        #issue attack command and sleep for an appropriate amount of time
        self.agent_host.sendCommand(attack_command)
        if move_command == 'move 0':
            time.sleep(7) #freeze in place for 7 secs (change this when we implement attacking)
        else:
            time.sleep(0.2)

        self.episode_step += 1

        # Get Observation
        print("getting a new observation")
        world_state = self.agent_host.getWorldState()
        
        for error in world_state.errors:
            print("Error:", error.text)

        print("about to get observation and store it")
        self.obs = self.get_observation(world_state) 
        print("updated self.obs")


        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def reset(self):
        """
        Resets the environment for the next episode.
        Returns
            observation: <np.array> flattened initial obseravtion
        """
        print("IM RESETTING")
        # Reset Malmo
        world_state = self.init_malmo(self.agent_host).getWorldState()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        print("Returns: ", self.returns)
        print("Steps: ", self.steps)

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs

    def get_observation(self, world_state):
        obs = np.zeros((4,432,240))     # observation_space

        if world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            for frame in world_state.video_frames:
                # Render the color map - still a bit laggy
                self.drawer.processFrame(frame)
                self.root.update()

                # Get observation from color map frame
                if frame.frametype == MalmoPython.FrameType.COLOUR_MAP:    
                    if world_state.number_of_observations_since_last_state > 0:
                        msg = world_state.observations[-1].text
                        observations = json.loads(msg)

                        # Rotate observation with orientation of agent 
                        # - added this back in since center pixel RGB val wasn't updating
                        yaw = observations['Yaw']

                        if yaw >= 225 and yaw < 315:
                            obs = np.rot90(obs, k=1, axes=(1, 2))
                        elif yaw >= 315 or yaw < 45:
                            obs = np.rot90(obs, k=2, axes=(1, 2))
                        elif yaw >= 45 and yaw < 135:
                            obs = np.rot90(obs, k=3, axes=(1, 2))
                        break
                

        time.sleep(1)
        self.drawer.reset()
        #print(obs)
        print("about to return from obs")

        return obs

    def init_malmo(self, agent_host):
        #Set up mission
        my_mission = MalmoPython.MissionSpec( getXML(MAX_EPISODE_STEPS=100,SIZE=10,N_TREES=2), True)

        #Record mission
        my_mission_record = MalmoPython.MissionRecordSpec()
        if not os.path.exists(os.path.sep.join([os.getcwd(), 'recordings'])):
            os.makedirs(os.path.sep.join([os.getcwd(), 'recordings']))
      
        my_mission.requestVideoWithDepth(432, 240)
        my_mission.setViewpoint(0)

        #Begin mission
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)
                    continue
        return agent_host

    def harvestWood(self):
        print("HARVESTING")
        time.sleep(1)
        self.agent_host.sendCommand("pitch 0")
        self.agent_host.sendCommand( "move 0.7")
        self.agent_host.sendCommand("attack 1")
        time.sleep(2)  #give it 2 seconds to collect wood
        self.agent_host.sendCommand( "move 0") #then freeze it and set attack to 0
        self.agent_host.sendCommand("attack 0")
        time.sleep(1)
        print("DONE HARVESTING")

    # Modified checkForWood to return appropriate actions to take if agent sees wood.
    def checkForWood(self, world_state, action):
        print("checking for wood")
        for f in (world_state.video_frames):
            if f.frametype == MalmoPython.FrameType.COLOUR_MAP:
                
                print("found a colormap frame")
                frame = f.pixels
                byte_list = list(frame)
                flat_img_array = np.array(byte_list)
                img_array = flat_img_array.reshape(240, 432, 3)
                center_y, center_x = 119, 215 #this is (240/2 - 1, 432/2 - 1)
                R,B,G = img_array[center_y][center_x][0], img_array[center_y][center_x][1], img_array[center_y][center_x][2]
                print("R,B,G: {}, {}, {}".format(R,B,G))
                if (R,B,G) == colors['wood']:
                    #freeze if we see wood(change this to moving + attacking later)
                    print("FOUND WOOD!")
                    return ("move 0", "turn 0", "attack 0")

        #otherwise, return the commands from action
        move_command = 'move {}'.format(action[0])
        turn_command = 'turn {}'.format(action[1])
        attack_command = 'turn {}'.format(action[2])

       
        return(move_command, turn_command, attack_command)

    def getCenterRGB(self):
        print()
        
        

if __name__ == '__main__':
    ModelCatalog.register_custom_model('pixelview_model', PixelViewModel)

    ray.init()
    trainer = ppo.PPOTrainer(env=SurvivAI, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0,           # We aren't using parallelism
        'model': {
            'custom_model': 'pixelview_model',
            'custom_model_config': {}
        }
    })
    

    while True:
       print(trainer.train())