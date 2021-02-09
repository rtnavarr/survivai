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

COLOURS = {'wood': (0, 93, 162), 'leaves':(232, 70, 162), 'grass':(46, 70, 139)}


def generateXZ(quadrant,SIZE):
    x = randint(1,SIZE)
    z = randint(1,SIZE)
    if quadrant == 0:
        return x,z
    if quadrant == 1:
        return x,-z
    if quadrant == 2:
        return -x,z
    return -x,-z

def getXML(MAX_EPISODE_STEPS, SIZE, N_TREES):

    my_xml = ""

    #generate 1 randomly-placed log per quadrant
    for i in range(4):
        x,z = generateXZ(i,SIZE)
        my_xml += "<DrawBlock x='{}' y='2' z='{}' type='log' />".format(x,z)

    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <About>
                    <Summary>Survivai Agent</Summary>
                </About>
                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>1000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='{}' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZE, SIZE, -SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='brick_block'/>".format(-SIZE-1, SIZE+1, -SIZE-1, SIZE+1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            my_xml + \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>Survivai</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="0" yaw="0"/>'.format(0, 0) + \
                        '''
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_axe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <ContinuousMovementCommands turnSpeedDegs="180"/>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>800</Width>
                            <Height>500</Height>
                        </ColourMapProducer>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <AgentQuitFromTouchingBlockType>
                            <Block type="log"/>
                        </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    
def init_malmo(agent_host):
    #Set up mission
    my_mission = MalmoPython.MissionSpec( getXML(100,10,2), True)

    #Record mission
    my_mission_record = MalmoPython.MissionRecordSpec()

    #my_mission.setDestination("recordings//survivai.tgz")
    my_mission_record.setDestination(os.path.sep.join([os.getcwd() + '/recordings', 'recording_' + str(int(time.time())) + '.tgz']))
    my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)

    my_mission.requestVideoWithDepth(800, 500)
    my_mission.setViewpoint(1)

    #Begin mission
    max_retries = 3
    #my_clients = MalmoPython.ClientPool()
    #my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

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

def get_observation(world_state):
        obs = np.zeros((4,800,500))

        agent_host.sendCommand("turn 0.1")
        time.sleep(1.0)
        agent_host.sendCommand("move 0.2")

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if len(world_state.video_frames):
                for frame in world_state.video_frames:
                    if frame.channels == 4:
                        break
                if frame.channels == 4:
                    pixels = world_state.video_frames[0].pixels
                    obs = np.reshape(pixels, (4, 800, 500))
                    break
                else:
                    pass
                    print('no depth found')
        print(obs)
        return obs

def train(agent_host):
    # Setup Malmo and get observation
    agent_host = init_malmo(agent_host)
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

            agent_host.sendCommand("move 0.5")
            time.sleep(2)

            for error in world_state.errors:
                print("\nError:", error.text)
    obs = get_observation(world_state)

    # Run episode
    print("\nRunning")
    while world_state.is_mission_running:
        # Replace with get action, take step, and sleep
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

        agent_host.sendCommand("move 0.5")
        time.sleep(2)

        for error in world_state.errors:
            print("Error:",error.text)

        for f in world_state.video_frames:
            if f.frametype == MalmoPython.FrameType.COLOUR_MAP:
                center_x = 400
                center_y = 250
                if (f.pixels[center_x*center_y], f.pixels[center_x*center_y*2], f.pixels[center_x*center_y*3]) == COLOURS['wood']:
                    print("found wood?")
                print('R:' + str(f.pixels[center_x*center_y]))
                print('G:' + str(f.pixels[center_x*center_y*2]))
                print('B:' + str(f.pixels[center_x*center_y*3]))

    print()
    print("Mission ended")




if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)
    train(agent_host)
    #call train on agent_host here!