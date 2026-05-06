import numpy as np

tile = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
tile2 = [100, 111, 222, 133, 144, 155, 166, 177, 188, 199, 200, 12, 13, 14, 15, 16]

# the conversion function which produces a single real value to represent a tile
def f(tile): # luminance only captures the intensity of the tile/image in grayscale channel
    arr = np.array(tile)
    mean = arr.mean()
    std = arr.std()
    return std / mean

print(f(tile))
print(f(tile2))