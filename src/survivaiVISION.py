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

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    from Tkinter import *
else:
    from tkinter import *
from PIL import ImageTk
from PIL import Image

video_width = 432
video_height = 240
WIDTH = video_width
HEIGHT = video_height

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
        if frame.frametype == MalmoPython.FrameType.COLOUR_MAP:
            # Use the centre slice of the colourmap to create a panaramic image
            # First create image from this frame:
            cmap = Image.frombytes('RGB', (video_width, video_height), bytes(frame.pixels))
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
        ###############################
        # Depth Map portion of helper #
        ###############################
        # elif frame.frametype == MalmoPython.FrameType.DEPTH_MAP:
        #     # Use the depth map to create a "radar" - take just the centre point of the depth image,
        #     # and use it to add a "blip" to the radar screen.

        #     # Set up some drawing params:
        #     size = min(WIDTH, HEIGHT)
        #     scale = old_div(size, 20.0)
        #     angle = frame.yaw * math.pi / 180.0
        #     cx = old_div(size, 2)
        #     cy = cx

        #     # Draw the sweeping line:
        #     points = [cx, cy, cx + 10 * scale * math.cos(angle), cy + 10 * scale * math.sin(angle), cx + 10 * scale * math.cos(self._last_angle), cy + 10 * scale * math.sin(self._last_angle)]
        #     self._last_angle = angle
        #     self._segments.append(self._canvas.create_polygon(points, width=0, fill="#004410"))

        #     # Get the depth value from the centre of the map:
        #     mid_pix = 2 * video_width * (video_height + 1)  # flattened index of middle pixel
        #     depth = scale * struct.unpack('f', bytes(frame.pixels[mid_pix:mid_pix + 4]))[0]   # unpack 32bit float

        #     # Draw the "blip":
        #     x = cx + depth * math.cos(angle)
        #     y = cy + depth * math.sin(angle)
        #     self._dots.append((self._canvas.create_oval(x - 3, y - 3, x + 3, y + 3, width=0, fill="#ffa930"), self._current_frame))

        #     # Fade the lines and the blips:
        #     for i, seg in enumerate(self._segments):
        #         fillstr = "#{0:02x}{1:02x}{2:02x}".format(0, int((self._line_fade - len(self._segments) + i) * (old_div(255.0, float(self._line_fade)))), 0)
        #         self._canvas.itemconfig(seg, fill=fillstr)
        #     if len(self._segments) >= self._line_fade:
        #         self._canvas.delete(self._segments.pop(0))

        #     for i, dot in enumerate(self._dots):
        #         brightness = self._blip_fade - (self._current_frame - dot[1])
        #         if brightness < 0:
        #             self._canvas.delete(dot[0])
        #         else:
        #             fillstr = "#{0:02x}{1:02x}{2:02x}".format(100, int(brightness * (old_div(255.0, float(self._blip_fade)))), 80)
        #             self._canvas.itemconfig(dot[0], fill=fillstr)
        #         self._dots = [dot for dot in self._dots if self._current_frame - dot[1] <= self._blip_fade]
        #     self._current_frame += 1