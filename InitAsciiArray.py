## The purpose of this file is to initialize an array of ASCII characters as a glyph codebook
## as done in 'Fast Rendering of Image Mosaics and ASCII Art' (2015)

# We must select a monospace font and enumerate all characters from that font that we want to use.
# We must then encode the graphical representation of each character for [image tile] -- [ASCII tile] comparisons.
# The map<enum, tile> will allow us to render the ASCII tiles just once.

from PIL import Image, ImageDraw, ImageFont

# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
def generate_images(width, height): # w and h given in pixels
    img = Image.new('L', (width, height), color=255) # L indicates grayscale. You know, like grayscaLe.
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("consola.ttf", 26) # on Windows check 'c:\%windir%\fonts' to see what's available (choose a monospace font)
    text = 'A' #TODO: this should be an array of all ASCII chars we want to use
    draw.text((0, 0), text, fill=(0), font=font) # start position, content, fill color (white), font
    img.show()

generate_images(14, 22) # manually adjusted to match font size of 26.
#TODO need some way to calculate tile-size to font-size ratio
#TODO add third argument that specifies start and end index of ASCII char set we want to use

#TODO After we generate the images, we need to decompose each and create the map of <ASCII index, image data>