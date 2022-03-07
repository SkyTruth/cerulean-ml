import numpy as np
import dask

# Hard Neg is overloaded with overlays but they shouldn't be exported during annotation
# Hard Neg is just a class that we will use to measure performance gains metrics
class_mapping_photopea = {"Infrastructure": (0,0,255),
                "Natural Seep":(0,255,0),
                "Coincident Vessel":(255,0,0),
                "Recent Vessel":(255,255,0),
                "Old Vessel": (255,0, 255),
                "Ambiguous": (255,255,255),
                "Hard Negatives":(0,255,255)}

class_mapping_coco = {"Infrastructure": 1,
                "Natural Seep":2,
                "Coincident Vessel":3,
                "Recent Vessel":4,
                "Old Vessel": 5,
                "Ambiguous": 6,
                "Hard Negatives":0}

class_mapping_coco_inv = {1:"Infrastructure",
                2:"Natural Seep",
                3:"Coincident Vessel",
                4:"Recent Vessel",
                5:"Old Vessel",
                6:"Ambiguous",
                0:"Hard Negatives"}

def pad_l_total(chip_l, img_l):
    """find the total amount of padding that needs to occur 
    for an array with length img_l to be tiled to a chip size with a length chip_l"""
    return chip_l* (1 - (img_l/chip_l - img_l//chip_l))

def reshape_split(image: np.ndarray, kernel_size: tuple):
    """
    Adapted from https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    """
    if len(image.shape) ==2:
        image = np.expand_dims(image, axis=2)
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size
    pad_height = pad_l_total(tile_height, img_height)
    pad_width = pad_l_total(tile_width, img_width)
    pad_height_up = int(np.floor(pad_height/2))
    pad_height_down = int(np.ceil(pad_height/2))
    pad_width_up = int(np.floor(pad_width/2))
    pad_width_down = int(np.ceil(pad_width/2))
    image_padded = np.pad(image, ((pad_height_up, pad_height_down), (pad_width_up, pad_width_down), (0,0)), mode="constant", constant_values=0)
    img_height, img_width, channels = image_padded.shape
    tiled_array = image_padded.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    if tiled_array.shape[-1] == 4:
        return tiled_array.reshape(tiled_array.shape[0]*tiled_array.shape[1], tile_width,tile_height, 4)
    elif tiled_array.shape[-1] == 1:
        return tiled_array.reshape(tiled_array.shape[0]*tiled_array.shape[1], tile_width,tile_height)
    else:
        return ValueError(f"The shape of the tiled array before simplification is {tiled_array.shape}, this needs correcting because the last dim should represent channels with either 1 channel for the image array or 4 for the label array.")

def save_tiles_from_3d(tiled_arr, img_fname, outdir):
    """
    Dask gives a linear speedup for saving out png files. This timing indiciates it would take .92 
    hours to tile the 500 background images with 4 cores running. `500 images * 6.67 seconds / 60 / 60 = .92`

    Since we have an image for each instance, this time could take a while if we saved out an instance 
    for each image. but since we have the coco dataset format we can keep annotations in memory and 
    not save out annotation images.
    """
    tiles_n, _, _ = tiled_arr.shape
    lazy_results = []
    for i in range(tiles_n):
        fname = os.path.join(outdir, os.path.basename(os.path.dirname(img_fname))+f"_vv-image_tile_{i}.png")
        lazy_result = dask.delayed(skio.imsave)(fname,tiled_arr[i])
        lazy_results.append(lazy_result)
    results = dask.compute(*lazy_results)
    print(f"finished saving {tiles_n} images")
    
def rgbalpha_to_binary(arr, r,g,b):
    return np.logical_and.reduce([arr[:,:,0]==r,arr[:,:,1]==g,arr[:,:,2]==b])

def is_layer_of_class(arr, r,g,b):
    return np.logical_and.reduce([arr[:,:,0]==r,arr[:,:,1]==g,arr[:,:,2]==b]).any()

def get_layer_cls(arr, class_mapping_photopea, class_mapping_coco):
    if len(arr.shape) == 2:
        raise ValueError("This expects a 3D array with 4 channels representing a label array, got array with only 1 channel.")
    elif len(arr.shape) == 3 and arr.shape[-1] == 4:
        for category in class_mapping_photopea.keys():
            if is_layer_of_class(arr, *class_mapping_photopea[category]):
                return class_mapping_coco[category]
        return 0 # no category matches, all background label
    else:
        raise ValueError("Check the array to make sure it is a label array with 4 channels for rgb alpha.")