import numpy as np

# Hard Neg is overloaded with overlays but they shouldn't be exported during annotation
# Hard Neg is just a class that we will use to measure performance gains metrics
class_mapping = {"Infrastructure": (0,0,255),
                "Natural Seep":(0,255,0),
                "Coincident Vessel":(255,0,0),
                "Recent Vessel":(255,255,0),
                "Old Vessel": (255,0, 255),
                "Ambiguous": (255,255,255),
                "Hard Negatives":(0,255,255)}

def pad_l_total(chip_l, img_l):
    """find the total amount of padding that needs to occur 
    for an array with length img_l to be tiled to a chip size with a length chip_l"""
    return chip_l* (1 - (img_l/chip_l - img_l//chip_l))

def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width = image.shape
    tile_height, tile_width = kernel_size
    pad_height = pad_l_total(tile_height, img_height)
    pad_width = pad_l_total(tile_width, img_width)
    pad_height_up = int(np.floor(pad_height/2))
    pad_height_down = int(np.ceil(pad_height/2))
    pad_width_up = int(np.floor(pad_width/2))
    pad_width_down = int(np.ceil(pad_width/2))
    image_padded = np.pad(image, ((pad_height_up, pad_height_down), (pad_width_up, pad_width_down)), mode="constant", constant_values=0)
    img_height, img_width = image_padded.shape
    tiled_array = image_padded.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array