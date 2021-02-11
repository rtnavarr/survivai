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

from generate_map import generateXZ, getXML
import survivaiVISION
from survivaiVISION import draw_helper

COLOURS = {'wood': (0, 93, 162), 'leaves':(232, 70, 162), 'grass':(46, 70, 139)}


class SurvivAI(gym.Env):
    def __init__(self, env_config):
        # RLLib parameters
        # none yet
        
        self.obs_size = 5
        self.action_space = Box(np.array([-1,-1,-1]), np.array([1,1,1]), dtype=np.int32)
        self.observation_space = Box(0, 1, shape=(2 * self.obs_size * self.obs_size, ), dtype=np.float32)

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
  
        self.train()

    def train(self):
        # Setup Malmo and get observation
        self.agent_host = self.init_malmo(self.agent_host)
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.05)
            world_state = self.agent_host.getWorldState()

            for error in world_state.errors:
                print("\nError:", error.text)
    
        obs = self.get_observation(world_state)
        self.agent_host.sendCommand( "turn 0.05" )

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

            for f in world_state.video_frames:
                if f.frametype == MalmoPython.FrameType.COLOUR_MAP:
                    frame = f.pixels
                    byte_list = list(frame)
                    flat_img_array = np.array(byte_list)
                    img_array = flat_img_array.reshape(240, 432, 3)
                    center_y, center_x = 119, 215 #this is (240/2 - 1, 432/2 - 1)
                    R,B,G = img_array[center_y][center_x][0], img_array[center_y][center_x][1], img_array[center_y][center_x][2]
                    print("R,B,G = {}, {}, {}".format(str(R), str(B), str(G)))
                    if (R,B,G) == (162, 0, 93):
                        print("FOUND WOOD!")
                        self.agent_host.sendCommand("turn 0.0")
                        time.sleep(2)
                        self.agent_host.sendCommand("turn 0.05")

        time.sleep(1)
        self.drawer.reset()
        print()
        print("Mission ended")
    

    def get_observation(self, world_state):
        obs = np.zeros((4,432,240))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

            if world_state.number_of_video_frames_since_last_state > 0:
                self.drawer.processFrame(world_state.video_frames[-1])
                self.root.update()

            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if len(world_state.video_frames):
                for frame in world_state.video_frames:
                    if frame.channels == 4:
                        break
                if frame.channels == 4:
                    pixels = world_state.video_frames[0].pixels
                    obs = np.reshape(pixels, (4, 432, 240))
                    
                    if world_state.number_of_observations_since_last_state > 0:
                        # First we get the json from the observation API
                        msg = world_state.observations[-1].text
                        observations = json.loads(msg)
                        # Rotate observation with orientation of agent
                        yaw = observations['Yaw']
                        if yaw >= 225 and yaw < 315:
                            obs = np.rot90(obs, k=1, axes=(1, 2))
                        elif yaw >= 315 or yaw < 45:
                            obs = np.rot90(obs, k=2, axes=(1, 2))
                        elif yaw >= 45 and yaw < 135:
                            obs = np.rot90(obs, k=3, axes=(1, 2))
                    
                    break
                else:
                    pass
                    print('no depth found')
        time.sleep(1)
        self.drawer.reset()
        #print(obs)
        return obs

    def init_malmo(self, agent_host):
        #Set up mission
        my_mission = MalmoPython.MissionSpec( getXML(100,10,2), True)

        #Record mission
        my_mission_record = MalmoPython.MissionRecordSpec()
        if not os.path.exists(os.path.sep.join([os.getcwd(), 'recordings'])):
            os.makedirs(os.path.sep.join([os.getcwd(), 'recordings']))
        my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recordings', 'recording_' + str(int(time.time())) + '.tgz']))
        my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)

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

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=SurvivAI, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    #while True:
    #    print(trainer.train())
    #SurvivAI(gym.Env)