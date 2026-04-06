## The key purpose of this file is to decompose an image into a single value which represents
## all tiles of the original image.

# Current approaches to consider:
#   Structural Similarity Index (SSIM) - Probably fastest and simplest. Primarily weights tile 'luminance'
#   Alignment-Insensitive Shape Similarity (AISS) - Probably more complex, would need to implement a B-Tree. Captures both luminance and structure.

from PIL import Image, ImageOps
import numpy as np

# The main calling function. Should return a single value (large int?) that represents the image
def decompose(image_file, rescale_size, tile_w, tile_h):
    # load image, grayscale, and downscale it
    img = Image.open(f'images/{image_file}')
    img = ImageOps.grayscale(img.resize((rescale_size, rescale_size)))

    #test
    img.show()
    tiles = get_tiles(img, tile_w, tile_h)
    tile_values = []
    for t in tiles:
        tile_values.append(f(t))
    return tile_values

    #print(tiles[len(tiles)//3]) # prints the values from somewhere around 1/3 through the image

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

def f(tile): # luminance only captures the intensity of the tile/image in grayscale channel
    return np.average(tile) / 255.0

image_name = 'moon.jpg'
tile_values = decompose(image_name, 576, 16, 24) # 576 evenly divisible by 16 and 24
for tv in tile_values:
    print(tv)
