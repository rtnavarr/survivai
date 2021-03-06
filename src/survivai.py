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
from scipy.stats import mode as mode

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
        self.action_space = Box(low=np.array([-1, -1, -1]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64)
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
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.KEEP_ALL_FRAMES )
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



        self.can_break = False
   
        # self.train()

    def train(self):
        # Setup Malmo and get observation
        self.agent_host = self.init_malmo(self.agent_host)
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

            for error in world_state.errors:
                print("\nError:", error.text)
    
        obs = self.get_observation(world_state)
        print(obs)
        time.sleep(0.1)

        # Run episode
        print("\nRunning")
        while world_state.is_mission_running:
            # Replace with get action, take step, and sleep
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

            if world_state.number_of_video_frames_since_last_state > 0:
                self.drawer.processFrame(world_state.video_frames[-1])
                self.root.update()

            for error in world_state.errors:
                print("Error:",error.text)

        time.sleep(1)
        self.drawer.reset()
        print("Mission ended")
    
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
        for i in range(len(action)):
            if i == 0:  # move
                self.agent_host.sendCommand("move " + str(action[i]))
            if i == 1:  # turn
                self.agent_host.sendCommand("turn " + str(action[i]))
            if i == 2 and action[i] >= 0:
                print("Stop moving and look for trees")
                self.agent_host.sendCommand("move 0.0")
                self.agent_host.sendCommand("turn 0.0")
                world_state = self.agent_host.getWorldState()
                obs = self.get_observation(world_state)
                self.checkForWood(world_state)
            time.sleep(0.1)
        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 
        
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

        # while world_state.is_mission_running:
        if world_state.is_mission_running: 
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')
            
            if len(world_state.video_frames):
                for frame in reversed(world_state.video_frames):

                    if frame.channels == 4:
                        pixels = frame.pixels
                        if len(pixels) == 414720:    # 4 * 432 * 240 => ok for reshaping
                            obs = np.reshape(pixels, (4, 432, 240))
                            self.drawer.showFrame(frame)
                            return obs
            else:
                print("No video frames")
        time.sleep(1)
        self.drawer.reset()

        return obs

    def init_malmo(self, agent_host):
        #Set up mission
        my_mission = MalmoPython.MissionSpec( getXML(MAX_EPISODE_STEPS=100,SIZE=10,N_TREES=4), True)

        #Record mission
        my_mission_record = MalmoPython.MissionRecordSpec()
        if not os.path.exists(os.path.sep.join([os.getcwd(), 'recordings'])):
            os.makedirs(os.path.sep.join([os.getcwd(), 'recordings']))
        # my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recordings', 'recording_' + str(int(time.time())) + '.tgz']))
        # my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)

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
        time.sleep(0.1)
        self.agent_host.sendCommand("pitch 0")
        self.agent_host.sendCommand( "move 0.7")
        self.agent_host.sendCommand("attack 1")
        time.sleep(2)  #give it 2 seconds to collect wood
        self.agent_host.sendCommand( "move 0") #then freeze it and set attack to 0
        self.agent_host.sendCommand("attack 0")
        time.sleep(1)
        print("DONE HARVESTING")

    def checkForWood(self, world_state):
        if world_state.video_frames:
            f = world_state.video_frames[-1]
            if f.frametype == MalmoPython.FrameType.COLOUR_MAP:
                frame = f.pixels
                byte_list = list(frame)
                flat_img_array = np.array(byte_list)
                img_array = flat_img_array.reshape(240, 432, 3)

                #Extract 4x4 box of (R,B,G), can also try enlarging this later
                top = (240//2) - 2
                bottom = (240//2) + 2
                left = (432//2) - 2
                right = (432//2) + 2
                center_box = img_array[top:bottom, left:right]
                
                #Generate counts for each unique color pixel in the box
                dict = {}
                for row in center_box:
                    for pixel in row:
                        tup = tuple(pixel)
                        try:
                            dict[tup] += 1
                        except KeyError:
                            dict[tup] = 1
                print(dict)

                #Act based on whether or not the majority of the pixels in the box are wood
                try:
                    numWoodPixels = dict[(162,0,93)] 
                    if numWoodPixels >= 8: #wood is the majority "vote" pixel in a 4x4 box, change 8 to something else if box dims change
                        print("window's majority is wood, should attack")
                        self.agent_host.sendCommand("turn 0.0") #stop turning if we see wood
                        self.harvestWood()
                        self.agent_host.sendCommand("attack 0")
                    else:
                        print("window's majority is not wood")
                except KeyError:
                    print("window didnt contain any wood")

                '''
                #Old method of checking just the center pixel
                center_y, center_x = 119, 215 #this is (240/2 - 1, 432/2 - 1)
                R,B,G = img_array[center_y][center_x][0], img_array[center_y][center_x][1], img_array[center_y][center_x][2]
                print(R,B,G)
                if (R,B,G) == colors['wood']:
                    self.agent_host.sendCommand("turn 0.0") #stop turning if we see wood
                    print("FOUND WOOD!")
                    self.harvestWood()
                    self.agent_host.sendCommand("attack 0")
                '''
    
    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Wood Collection')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value)) 
        
        

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
