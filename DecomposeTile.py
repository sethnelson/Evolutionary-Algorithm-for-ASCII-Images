import numpy as np

# # the conversion function which produces a single real value to represent a tile
# def f(tile): # luminance only captures the intensity of the tile/image in grayscale channel
#     arr = np.array(tile)
#     mean = arr.mean()
#     std = tile.std()
#     return mean + std

# the conversion function which produces a single real value to represent a tile
def f(tile): # luminance only captures the intensity of the tile/image in grayscale channel
    arr = np.array(tile, dtype=float)
    return arr.mean() / 255.0