## The purpose of this file is to initialize an array of ASCII characters as a glyph codebook
## as done in 'Fast Rendering of Image Mosaics and ASCII Art' (2015)

# We must select a monospace font and enumerate all characters from that font that we want to use.
# We must then encode the graphical representation of each character for [image tile] -- [ASCII tile] comparisons.
# The map<enum, tile> will allow us to render the ASCII tiles just once.

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from DecomposeTile import f

# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
def char_to_img(width, height, ascii_code): # w and h given in pixels
    img = Image.new('L', (width, height), color=0) # L indicates grayscale. You know, like grayscaLe.
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("consola.ttf", 29) # on Windows check 'c:\%windir%\fonts' to see what's available (choose a monospace font)
    text = chr(ascii_code)
    draw.text((0, 0), text, fill=(255), font=font) # start position, content, fill color (black), font
    #img.show()
    return img

def build_dict(tile_w, tile_h): #builds an ASCII dict of tiles (w x h)
    exception_list = [127, 129, 141, 143, 144, 157] # these entries are blank on https://www.ascii-code.com/
    char_dict = {}
    for n in range(32, 127):
        if n in exception_list:
            continue
        else:
            img = char_to_img(tile_w, tile_h, n)
            char_dict[n] = f(img)
    return char_dict

if __name__ == "__main__":
    exception_list = [127, 129, 141, 143, 144, 157]
    dict = build_dict(16, 24) # manually adjusted to match font size of 29 and have dimensions that are factors of 4
    for n in range(32, 127):
            if n in exception_list:
                continue
            else:
                v = dict[n]
                print(f"{n} : {v}")