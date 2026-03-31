from PIL import Image
import numpy as np
import os

# The main calling function. Should return a single value (large int?) that represents the image
def decompose(image_file, n):
    img = Image.open(f'images/{image_file}')
    img = img.resize((n, n))
    img.show()
    t = get_tiles(img, 4, 8)

# Breaks images into tiles and returns an array of tiles
# Need raw image, tile width and tile height
def get_tiles(image, M, N):
    # Source - https://stackoverflow.com/a/47581978
    # Posted by Nir, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-03-31, License - CC BY-SA 3.0

    img_bytes = np.array(image)
    tiles = [img_bytes[x:x+M,y:y+N] for x in range(0,img_bytes.shape[0],M) for y in range(0,img_bytes.shape[1],N)]

    print(tiles[len(tiles)//3]) # prints the values from somewhere around 1/3 through the image
    return tiles

# takes an image of any size and returns that image downscaled to the specified nxn dimension
def downscale(image, n):
    
    return 0

image_name = 'moon.jpg'
decompose(image_name, 256)
