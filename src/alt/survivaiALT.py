# Rev 1
# Author: Tuan

from __future__ import print_function
from __future__ import division
from future import standard_library

try:
    from malmo import MalmoPython
except:
    import MalmoPython


standard_library.install_aliases()
from builtins import bytes
from builtins import range
from builtins import object
from past.utils import old_div
import random
import time
import logging
import struct
import socket
import os
import sys
import errno
import json
import math
import malmoutils
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

malmoutils.fix_print()
agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)

# standard RGB format (red, green, blue) 
COLORS = {'wood' : (162, 0, 93), 'leaves': (162, 232, 70), 'grass':(139, 46, 70)}

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    from Tkinter import *
else:
    from tkinter import *
from PIL import ImageTk
from PIL import Image

video_width = 432
video_height = 240
WIDTH = 432
HEIGHT = 432 + video_height

root = Tk()
root.wm_title("Depth and ColourMap Example")
root_frame = Frame(root)
canvas = Canvas(root_frame, borderwidth=0, highlightthickness=0, width=WIDTH, height=HEIGHT, bg="black")
canvas.config( width=WIDTH, height=HEIGHT )
canvas.pack(padx=5, pady=5)
root_frame.pack()

class draw_helper(object):
    def __init__(self, canvas):
        self._canvas = canvas
        self.reset()
        self._line_fade = 9
        self._blip_fade = 100

    def reset(self):
        self._canvas.delete("all")
        self._dots = []
        self._segments = []
        self._panorama_image = Image.new('RGB', (WIDTH, video_height))
        self._panorama_photo = None
        self._image_handle = None
        self._current_frame = 0
        self._last_angle = 0

    def processFrame(self, frame):
        if frame.frametype == MalmoPython.FrameType.DEPTH_MAP:
            # Use the depth map to create a "radar" - take just the centre point of the depth image,
            # and use it to add a "blip" to the radar screen.

            # Set up some drawing params:
            size = min(WIDTH, HEIGHT)
            scale = old_div(size, 20.0)
            angle = frame.yaw * math.pi / 180.0
            cx = old_div(size, 2)
            cy = cx

            # Draw the sweeping line:
            points = [cx, cy, cx + 10 * scale * math.cos(angle), cy + 10 * scale * math.sin(angle), cx + 10 * scale * math.cos(self._last_angle), cy + 10 * scale * math.sin(self._last_angle)]
            self._last_angle = angle
            self._segments.append(self._canvas.create_polygon(points, width=0, fill="#004410"))

            # Get the depth value from the centre of the map:
            mid_pix = 2 * video_width * (video_height + 1)  # flattened index of middle pixel
            depth = scale * struct.unpack('f', bytes(frame.pixels[mid_pix:mid_pix + 4]))[0]   # unpack 32bit float

            # Draw the "blip":
            x = cx + depth * math.cos(angle)
            y = cy + depth * math.sin(angle)
            self._dots.append((self._canvas.create_oval(x - 3, y - 3, x + 3, y + 3, width=0, fill="#ffa930"), self._current_frame))

            # Fade the lines and the blips:
            for i, seg in enumerate(self._segments):
                fillstr = "#{0:02x}{1:02x}{2:02x}".format(0, int((self._line_fade - len(self._segments) + i) * (old_div(255.0, float(self._line_fade)))), 0)
                self._canvas.itemconfig(seg, fill=fillstr)
            if len(self._segments) >= self._line_fade:
                self._canvas.delete(self._segments.pop(0))

            for i, dot in enumerate(self._dots):
                brightness = self._blip_fade - (self._current_frame - dot[1])
                if brightness < 0:
                    self._canvas.delete(dot[0])
                else:
                    fillstr = "#{0:02x}{1:02x}{2:02x}".format(100, int(brightness * (old_div(255.0, float(self._blip_fade)))), 80)
                    self._canvas.itemconfig(dot[0], fill=fillstr)
                self._dots = [dot for dot in self._dots if self._current_frame - dot[1] <= self._blip_fade]
            self._current_frame += 1
        elif frame.frametype == MalmoPython.FrameType.COLOUR_MAP:
            # Use the centre slice of the colourmap to create a panaramic image
            # First create image from this frame:
            cmap = Image.frombytes('RGB', (video_width, video_height), bytes(frame.pixels))
            frame_array = np.array(list(frame.pixels))
            img_array = frame_array.reshape(240, 432, 3)
            center_x, center_y = 215, 119
            rgb_center = tuple([img_array[center_y][center_x][0], img_array[center_y][center_x][1], img_array[center_y][center_x][2]])
            if rgb_center == COLORS['wood']:
                print('on LOG')
            # Now crop just the centre slice:
            # left = (old_div(video_width, 2)) - 4
            # cmap = cmap.crop((left, 0, left + 8, video_height))
            cmap.load()
            # Where does this slice belong in the panorama?
            # x = int((int(frame.yaw) % 360) * WIDTH / 360.0)
            # Paste it in:
            self._panorama_image.paste(cmap)
            # self._panorama_image.paste(cmap, (x, 0, x + 8, video_height))
            # Convert to a photo for canvas use:
            self._panorama_photo = ImageTk.PhotoImage(self._panorama_image)
            # And update/create the canvas image:
            if self._image_handle is None:
                self._image_handle = canvas.create_image(old_div(WIDTH, 2), HEIGHT - (old_div(video_height, 2)), image=self._panorama_photo)
            else:
                canvas.itemconfig(self._image_handle, image=self._panorama_photo)

class Survivai():
    def __init__(self):  
        # Static Parameters
        self.size = 20
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'attack 1'  # Destroy block
        }

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
        my_xml += "<DrawBlock x='{}' y='2' z='{}' type='log' />".format(x,z) + \
                  "<DrawBlock x='{}' y='3' z='{}' type='log' />".format(x,z) + \
                  "<DrawBlock x='{}' y='3' z='{}' type='log' />".format(x,z)

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
                            "<DrawCuboid x1='{}' x2='{}' y1='{}' y2='1' z1='{}' z2='{}' type='stone'/>".format(-SIZE, SIZE, -SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='stone'/>".format(-SIZE-1, SIZE+1, -SIZE-1, SIZE+1) + \
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
                        <DepthProducer>
                            <Width>''' + str(video_width) + '''</Width>
                            <Height>''' + str(video_height) + '''</Height>
                        </DepthProducer>
                        <ColourMapProducer>
                            <Width>''' + str(video_width) + '''</Width>
                            <Height>''' + str(video_height) + '''</Height>
                        </ColourMapProducer>
                        <ObservationFromFullStats/>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <AgentQuitFromTouchingBlockType>
                            <Block type="log"/>
                        </AgentQuitFromTouchingBlockType>
                        <ContinuousMovementCommands turnSpeedDegs="20" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    

# Create default Malmo objects:
missionXML = getXML(100,10,2)

agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
drawer = draw_helper(canvas)

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

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')
agent_host.sendCommand( "turn 1" )
# Loop until mission ends:
while world_state.is_mission_running:
    print(".", end="")
    world_state = agent_host.getWorldState()
    if world_state.number_of_video_frames_since_last_state > 0:
        drawer.processFrame(world_state.video_frames[-1])
        root.update()
    for error in world_state.errors:
        print("Error:",error.text)

time.sleep(1)
drawer.reset()
print()
print("Mission ended")
# Mission has ended.