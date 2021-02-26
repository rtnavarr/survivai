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

input_width = 432
input_height = 240
output_width = 432
output_height = 240
display_width = 432
display_height = 240# + input_width


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
        
        root.update()
    
    def showFrame(self, frame):
        orig_image = Image.frombytes('RGB', (input_width, input_height), bytes(frame.pixels))
        output_frame = orig_image.resize((output_width, output_height),  Image.NEAREST)
        display = output_frame.resize((display_width, display_height), Image.BOX)
        c = output_frame.getcolors(input_width * input_height)
        if c:
            log_pixels = {color: count for count, color in c}
        else:
            log_pixels = {}
        display.load()
        self._panorama_image.paste(display, (0, 0, display_width, display_height))
        self._panorama_photo = ImageTk.PhotoImage(self._panorama_image)
        # And update/create the canvas image:
        if self._image_handle is None:
            self._image_handle = canvas.create_image(0, 0, image=self._panorama_photo, anchor='nw')
        else:
            canvas.itemconfig(self._image_handle, image=self._panorama_photo)
        root.update()
        out = log_pixels[(1, 57, 110)] if (1, 57, 110) in log_pixels else 0
        return (out, np.array(output_frame))
