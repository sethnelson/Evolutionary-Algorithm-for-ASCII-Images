## The key purpose of this file is to decompose an image into a single value which represents
## all tiles of the original image.

# Current approaches to consider:
#   Structural Similarity Index (SSIM) - Probably fastest and simplest. Primarily weights tile 'luminance'
#   Alignment-Insensitive Shape Similarity (AISS) - Probably more complex, would need to implement a B-Tree. Captures both luminance and structure.

from PIL import Image, ImageOps
import numpy as np

# The main calling function. Should return a single value (large int?) that represents the image
def decompose(image_file, n):
    # load image, grayscale, and downscale it
    img = Image.open(f'images/{image_file}')
    img = ImageOps.grayscale(img.resize((n, n)))

    #test
    img.show()
    tiles = get_tiles(img, 4, 8)
    print(tiles[len(tiles)//3]) # prints the values from somewhere around 1/3 through the image

# Breaks images into tiles and returns an array of tiles
# Need raw image, tile width and tile height
# Tile concept from 'Fast Rendering of Image Mosaics and ASCII Art (2015)
def get_tiles(image, M, N):
    # Source - https://stackoverflow.com/a/47581978
    # Posted by Nir, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-03-31, License - CC BY-SA 3.0
    img_bytes = np.array(image)
    tiles = [img_bytes[x:x+M,y:y+N] for x in range(0,img_bytes.shape[0],M) for y in range(0,img_bytes.shape[1],N)]
    return tiles

def luminance(image): # luminance only captures the intensity of the tile/image in grayscale channel
    img = Image.open(f'images/{image}')
    img = ImageOps.grayscale(img.resize((32, 32)))
    return np.average(np.array(img)) / 255.0

image_name = 'moon.jpg'
#decompose(image_name, 32)
print(luminance(image_name))
