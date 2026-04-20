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
    luminance = arr / 255.0
    return luminance

def horizontal_filter(tile):
    cols = len(tile[0])
    rows = len(tile)
    filter_row = [1, 0, 1]
    filter = [[filter_row] * 3]
    print(filter)
    return 0


from DecomposeImage import *
if __name__ == "__main__":
    image_name = 'moon.jpg'
    img = Image.open(f'images/{image_name}')
    rescale_size = 574
    img = ImageOps.grayscale(img.resize((rescale_size, rescale_size)))
    tiles = get_tiles(img, 16, 24)
    print(tiles[len(tiles)//5])
    hfilter = horizontal_filter(tiles[len(tiles)//5])
    # tile_values = decompose(image_name, 576, 16, 24) # 576 evenly divisible by 16 and 24
    # onetile = tile_values[len(tile_values) // 2]
    # print(onetile)