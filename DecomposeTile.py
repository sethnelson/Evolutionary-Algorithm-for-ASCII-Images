import numpy as np
from scipy.signal import convolve2d 
# # the conversion function which produces a single real value to represent a tile
# def f(tile): # luminance only captures the intensity of the tile/image in grayscale channel
#     arr = np.array(tile)
#     mean = arr.mean()
#     std = tile.std()
#     return mean + std

# Kernel Filters
VERTICAL = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=float)
                                                                                                                                                                                                                                                                                            
HORIZONTAL = np.array([                                                                                                                                                                                                                                                                       
    [-1, -1, -1],                                                                                                                                                                                                                                                                             
    [ 0,  0,  0],                                                                                                                                                                                                                                                                             
    [ 1,  1,  1],                                                                                                                                                                                                                                                                             
], dtype=float)                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                            
DIAG_DOWN = np.array([                                                                                                                                                                                                                                                                        
    [ 0,  1,  1],                                                                                                                                                                                                                                                                             
    [-1,  0,  1],                                                                                                                                                                                                                                                                             
    [-1, -1,  0],                                                                                                                                                                                                                                                                             
], dtype=float)                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                            
DIAG_UP = np.array([                                                                                                                                                                                                                                                                          
    [-1, -1,  0],                                                                                                                                                                                                                                                                             
    [-1,  0,  1],                                                                                                                                                                                                                                                                             
    [ 0,  1,  1],                                                                                                                                                                                                                                                                             
], dtype=float)      

# Process through filters
def filter_response(arr, kernel):                                                                                                                                                                                                                                                             
    response = convolve2d(arr, kernel, mode="valid")                                                                                                                                                                                                                                          
    return np.abs(response).mean()                                                                                                                                                                                                                                                            

# instead of returning a single real value, return multiple values representing different
# properties of the tile                                                                                                                                                                                                                                                                      
def f(tile):                                                                                                                                                                                                                                                                                  
    arr = np.array(tile, dtype=float) / 255.0 # normalize 


    return np.array([
        arr.mean(), # luminance
        arr.std(),  # contrast
        filter_response(arr, VERTICAL),
        filter_response(arr, HORIZONTAL),
        filter_response(arr, DIAG_DOWN),
        filter_response(arr, DIAG_UP),
    ])

from DecomposeImage import *
if __name__ == "__main__":
    image_name = 'moon.jpg'
    img = Image.open(f'images/{image_name}')
    rescale_size = 574
    img = ImageOps.grayscale(img.resize((rescale_size, rescale_size)))
    tiles = get_tiles(img, 16, 24)