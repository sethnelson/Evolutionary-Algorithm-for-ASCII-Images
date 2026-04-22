## The purpose of this file is to initialize an array of ASCII characters as a glyph codebook
## as done in 'Fast Rendering of Image Mosaics and ASCII Art' (2015)

# We must select a monospace font and enumerate all characters from that font that we want to use.
# We must then encode the graphical representation of each character for [image tile] -- [ASCII tile] comparisons.
# The map<enum, tile> will allow us to render the ASCII tiles just once.

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from DecomposeTile import f
from DecomposeImage import get_tiles

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
    # exception_list = [127, 129, 141, 143, 144, 157] # these entries are blank on https://www.ascii-code.com/
    # 70 character ASCII ramp
    # ascii_ramp_codes = [
    #     36, 64, 66, 37, 56, 38, 87, 77, 35, 42,
    #     111, 97, 104, 107, 98, 100, 112, 113, 119, 109,
    #     90, 79, 48, 81, 76, 67, 74, 85, 89, 88,
    #     122, 99, 118, 117, 110, 120, 114, 106, 102, 116,
    #     47, 92, 124, 40, 41, 49, 123, 125, 91, 93,
    #     63, 45, 95, 43, 126, 60, 62, 105, 33, 32,
    #     108, 73, 59, 58, 44, 34, 94, 39, 96, 41
    # ]
    ascii_ramp_codes = [
        64, 35, 87, 36, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
        63, 33, 97, 98, 99, 59, 58, 43, 61, 45, 44, 46, 95
    ]
    # ascii_ramp_codes = [
    #     64, 37, 35, 42, 43, 61, 45, 58, 46
    # ]
    char_dict = {}
    for n in ascii_ramp_codes:
        # if n in exception_list:
        #     continue
        # else:
        img = char_to_img(tile_w, tile_h, n)
        tiles = get_tiles(img, tile_w//2, tile_h//2)
        tile_vals = []
        for t in tiles:
             tile_vals.append(f(t))
        char_dict[n] = tile_vals
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