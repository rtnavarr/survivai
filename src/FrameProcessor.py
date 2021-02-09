
try:
    from malmo import MalmoPython
except:
    import MalmoPython
from past.utils import old_div
from tkinter import *

from PIL import Image
from PIL import ImageTk
import numpy as np


input_width = 5
input_height = 5
output_width = 20
output_height = 20
display_width = 500
display_height = 500# + input_width

root = Tk()
root.wm_title("Depth and ColourMap Example")
root_frame = Frame(root)
canvas = Canvas(root_frame, borderwidth=0, highlightthickness=0, width=display_width, height=display_height, bg="black")
canvas.config( width=display_width, height=display_height )
canvas.pack(padx=5, pady=5)
root_frame.pack()

# https://github.com/microsoft/malmo/blob/c3d129721c5a2f7c0eac274836f113f4c7ae4205/Malmo/samples/Python_examples/radar_test.py
class draw_helper(object):
    def __init__(self):
        self._canvas = canvas
        self.reset()
        self._line_fade = 9
        self._blip_fade = 100

    def reset(self):
        self._canvas.delete("all")
        self._dots = []
        self._segments = []
        self._panorama_image = Image.new('RGB', (display_width, display_height))
        self._panorama_photo = None
        self._image_handle = None
        self._current_frame = 0
        self._last_angle = 0


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