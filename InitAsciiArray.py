## The purpose of this file is to initialize an array of ASCII characters as a glyph codebook
## as done in 'Fast Rendering of Image Mosaics and ASCII Art' (2015)

# We must select a monospace font and enumerate all characters from that font that we want to use.
# We must then encode the graphical representation of each character for [image tile] -- [ASCII tile] comparisons.
# The map<enum, tile> will allow us to render the ASCII tiles just once.

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
def generate_images(width, height, ascii_code): # w and h given in pixels
    img = Image.new('L', (width, height), color=255) # L indicates grayscale. You know, like grayscaLe.
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("consola.ttf", 29) # on Windows check 'c:\%windir%\fonts' to see what's available (choose a monospace font)
    text = 'A' #TODO: this should be an array of all ASCII chars we want to use (32-255) except [127, 129, 141, 143, 144, 157]
    draw.text((0, 0), text, fill=(0), font=font) # start position, content, fill color (white), font
    img.show()
    return img

#? Idea for how we can create images of all ASCII chars
# exception_list = [127, 129, 141, 143, 144, 157] # these entries are blank on https://www.ascii-code.com/
# for n in range(32, 256):
#     if n in exception_list:
#         continue
#     generate_images(16, 24, n)

img = generate_images(16, 24, 0) # manually adjusted to match font size of 29 and have dimensions that are factors of 4
print(np.average(np.array(img)) / 255.0) # for now we're going to use a raw luminance value for comparing tiles. T
#TODO need some way to calculate tile-size to font-size ratio
#TODO add third argument that specifies start and end index of ASCII char set we want to use

#TODO After we generate the images, we need to decompose each and create the map of <ASCII index, image data