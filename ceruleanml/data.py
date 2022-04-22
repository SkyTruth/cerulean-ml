import json
import os
import zipfile
from datetime import datetime
from io import BytesIO
from shutil import copy
from typing import Any, List, Optional, Tuple

import dask
import distancerasters as dr
import fiona
import httpx
import numpy as np
import rasterio
import skimage.io as skio
import skimage.transform
from pycococreatortools import pycococreatortools
from rasterio import transform
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image, reshape_as_raster
from rasterio.vrt import WarpedVRT

# Hard Neg is overloaded with overlays but they shouldn't be exported during annotation
# Hard Neg is just a class that we will use to measure performance gains metrics
class_mapping_photopea = {
    "Infrastructure": (0, 0, 255),
    "Natural Seep": (0, 255, 0),
    "Coincident Vessel": (255, 0, 0),
    "Recent Vessel": (255, 255, 0),
    "Old Vessel": (255, 0, 255),
    "Ambiguous": (255, 255, 255),
    "Hard Negatives": (0, 255, 255),
}

class_mapping_coco = {
    "Infrastructure": 1,
    "Natural Seep": 2,
    "Coincident Vessel": 3,
    "Recent Vessel": 4,
    "Old Vessel": 5,
    "Ambiguous": 6,
    "Hard Negatives": 0,
}

class_mapping_coco_inv = {
    1: "Infrastructure",
    2: "Natural Seep",
    3: "Coincident Vessel",
    4: "Recent Vessel",
    5: "Old Vessel",
    6: "Ambiguous",
    0: "Hard Negatives",
}


def pad_l_total(chip_l: int, img_l: int):
    """Find the total amount of padding that needs to occur
    for an array.

    Args:
        chip_l (int): The length of the tile
        img_l (int): The big image length that needs to be tiled

    Returns:
        float: The average padding that should occur on either side.
            This is a float, and should be rounded up or down on either side.
    """
    return chip_l * (1 - (img_l / chip_l - img_l // chip_l))


def reshape_split(image: np.ndarray, kernel_size: tuple):
    """Takes a large image and tile size and pads the image with zeros then
        splits the 2D image into a 3D tiled stack of images.

    Adapted from https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7

    Args:
        image (np.ndarray): The big array representing a Sentinel-1 VV scene
            or a label layer from photopea.
            Can have any number of channels but it must be shaped like (H, W, Channels)
        kernel_size (tuple): The size of a binary tile. (H, W)

    Returns:
        np.ndarray: A numpy array shaped like (number of tiles, H, W)
    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size
    pad_height = pad_l_total(tile_height, img_height)
    pad_width = pad_l_total(tile_width, img_width)
    pad_height_up = int(np.floor(pad_height / 2))
    pad_height_down = int(np.ceil(pad_height / 2))
    pad_width_up = int(np.floor(pad_width / 2))
    pad_width_down = int(np.ceil(pad_width / 2))
    image_padded = np.pad(
        image,
        ((pad_height_up, pad_height_down), (pad_width_up, pad_width_down), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    img_height, img_width, channels = image_padded.shape
    tiled_array = image_padded.reshape(
        img_height // tile_height,
        tile_height,
        img_width // tile_width,
        tile_width,
        channels,
    )
    tiled_array = tiled_array.swapaxes(1, 2)

    return tiled_array.reshape(
        tiled_array.shape[0] * tiled_array.shape[1],
        tile_width,
        tile_height,
        tiled_array.shape[-1],
    )


def save_tiles_from_3d(tiled_arr: np.ndarray, img_fname: str, outdir: str):
    """Saves tiles from a 3D array from data.reshape_split using a template
    string and outdir.

    Dask gives a linear speedup for saving out png files. This timing
    indicates it would take .92 hours to tile the 500 background images with
    4 cores running. `500 images * 6.67 seconds / 60 / 60 = .92`

    Since we have an image for each instance, this time could take a while
    if we saved out an instance for each image. but since we have the coco
    dataset format we can keep annotations in memory and not save out annotation
    images.

    Args:
        tiled_arr (np.ndarray): A 3D array shaped like (number of tiles, H, W)
        img_fname (str): the template string to identify the image tiles. Has a
            unique integer id at the end, starting from zero.
        outdir (str): The directory to save img tiles.
    """
    tiles_n, _, _, _ = tiled_arr.shape
    lazy_results = []
    for i in range(tiles_n):
        fname = os.path.join(
            outdir,
            os.path.basename(os.path.dirname(img_fname))
            + f"_vv-image_local_tile_{i}.tif",
        )
        lazy_result = dask.delayed(skio.imsave)(
            fname, tiled_arr[i], "tifffile", False
        )  # don't check contrast
        lazy_results.append(lazy_result)
    dask.compute(*lazy_results)
    print(f"finished saving {tiles_n} images")


def copy_whole_images(img_list: List, outdir: str):
    """Copy whole images from a directory (mounted gcp bucket) to another directory.

    Dask gives a linear speedup for saving out png files. This timing
    indicates it would take X hours to copy the X background images with
    4 cores running.

    With the coco format we can keep annotations in memory and not save out annotation
    images.

    Args:
        img_list (list): List of image file paths in mounted gcp dir (or regular dir)
        img_fname (str): the template string. original image fname is use dto id the whole images.
        outdir (str): The directory to save img tiles.
    """
    lazy_results = []
    for i in range(len(img_list)):
        out_fname = os.path.join(
            outdir, os.path.basename(os.path.dirname(img_list[i])) + "_Background.png"
        )
        in_fname = img_list[i]
        lazy_result = dask.delayed(copy)(in_fname, out_fname)
        lazy_results.append(lazy_result)
    dask.compute(*lazy_results)
    print(f"finished saving {len(img_list)} images")


def rgbalpha_to_binary(arr: np.ndarray, r: int, g: int, b: int):
    """Converts a label layer from photopea to a binary 2D ndarray.

    Args:
        arr (np.ndarray): The 3D numpy ndarray
        r (int): red integer id from class_mapping_photopea
        g (int): green integer id from class_mapping_photopea
        b (int): blue integer id from class_mapping_photopea

    Returns:
        np.ndarray: the binary array
    """
    return np.logical_and.reduce(
        [arr[:, :, 0] == r, arr[:, :, 1] == g, arr[:, :, 2] == b]
    )


def is_layer_of_class(arr, r, g, b):
    """Checks class of a label layer from photopea

    Args:
        arr (np.ndarray): The 3D numpy ndarray
        r (int): red integer id from class_mapping_photopea
        g (int): green integer id from class_mapping_photopea
        b (int): blue integer id from class_mapping_photopea

    Returns:
        bool: True if any of the class is in the layer.
    """
    return rgbalpha_to_binary(arr, r, g, b).any()


def get_layer_cls(
    arr: np.ndarray,
    class_mapping_photopea: dict = class_mapping_photopea,
    class_mapping_coco: dict = class_mapping_coco,
):
    """Returns the integer class id of the instance layer.

    Args:
        arr (np.ndarray): A 3D array with 4 channels
        class_mapping_photopea (dict): The class mapping from RGB values to
        class_mapping_coco (dict): _description_

    Raises:
        ValueError: raises an error if the array isn't formatted as it should
            be from photopea label export.

    Returns:
        _type_: integer id for the class as defined by the class_mapping_coco dict.
    """
    if len(arr.shape) == 3 and arr.shape[-1] == 4:
        for category in class_mapping_photopea.keys():
            if is_layer_of_class(arr, *class_mapping_photopea[category]):
                return class_mapping_coco[category]
        return 0  # no category matches, all background label
    else:
        raise ValueError(
            "Check the array to make sure it is a label array with 4 channels for rgb alpha."
        )


class COCOtiler:
    def __init__(self, img_dir: str, coco_output: dict):
        self.instance_id = 0
        self.global_tile_id = 0
        self.global_increment = 0
        self.big_image_id = 0
        self.coco_output = coco_output
        self.img_dir = img_dir

        self.s1_scene_id: Optional[str] = None
        self.s1_bounds: Optional[List[float]] = None
        self.s1_image_shape: Optional[Tuple[int, int]] = None
        self.s1_gcps: Optional[List[Any]] = None
        self.s1_crs: Optional[Any] = None

    def save_background_img_tiles(
        self,
        scene_id: str,
        layer_paths: List[str],
        aux_datasets: List[str] = [],
        **kwargs,
    ):
        """Save background image tiles with additional optional datasets (vector or ship_density)
        The output background tiles are either 3 or 4 channel images.
        This means there needs to be a minimum of 2 and a maximum of 3 auxiliary datasets.

        Args:
            scene_id (str): The originating scene_id for the background and annotations.
            layer_paths (List[str]): List of path in a scene folder corresponding to Background.png, Layer 1.png, etc. Order matters.
            aux_datasets (List[str], optional): List of paths pointing to auxiliary vector files to include in tiles OR ship_density. 55km is the range by default. Defaults to [].

        Raises:
            ValueError: Error if original source imagery is not VV polarization.
        """

        self.s1_scene_id = scene_id
        (
            self.s1_bounds,
            self.s1_image_shape,
            self.s1_gcps,
            self.s1_crs,
        ) = fetch_sentinel1_reprojection_parameters(scene_id)

        # saving vv image tiles (Background layer)
        img_path = layer_paths[0]

        with rasterio.open(img_path) as src:
            profile = src.profile.copy()
            profile["driver"] = "GTiff"
            profile["crs"] = self.s1_crs
            profile["gcps"] = self.s1_gcps

            with MemoryFile() as mem:
                with mem.open(**profile) as m:
                    ar = src.read()
                    new_ar = np.zeros(ar.shape, dtype=ar.dtype)
                    cmap = src.colormap(1)
                    for k, v in cmap.items():
                        new_ar[0, ar[0] == k] = v[0]
                    m.write(new_ar)
                    m.colorinterp = [ColorInterp.gray]
                    gcps_transform = transform.from_gcps(self.s1_gcps)
                    with WarpedVRT(
                        m,
                        src_crs=self.s1_crs,
                        src_transform=gcps_transform,
                        add_alpha=False,
                    ) as vrt_dst:
                        # arr is (c, h, w)
                        arr = vrt_dst.read(
                            out_shape=(vrt_dst.count, *self.s1_image_shape),
                            out_dtype="uint8",
                        )
                        assert arr.shape[1:] == self.s1_image_shape

        # Make sure there are channels
        arr = reshape_as_image(arr)

        # Handle aux dataset per scene
        if aux_datasets:
            aux_dataset_channels = self.handle_aux_datasets(
                aux_datasets,
                self.s1_scene_id,
                self.s1_bounds,
                self.s1_image_shape,
                **kwargs,
            )

            # append as channels to arr
            arr = np.concatenate([arr, aux_dataset_channels], axis=2)

        tiled_arr = reshape_split(arr, (512, 512))
        if "Background" in str(img_path):  # its the vv image
            save_tiles_from_3d(tiled_arr, img_path, self.img_dir)
        else:
            raise ValueError(f"The layer {img_path} is not a VV image.")

    def copy_background_images(self, class_folders: List[str]):
        fnames_vv = []
        for f in class_folders:
            # TODO: fix types
            fnames_vv.extend(list(f.glob("**/Background.png")))  # type: ignore
        copy_whole_images(fnames_vv, self.img_dir)

    def create_coco_from_photopea_layers(self, scene_id: str, layer_pths: List[str]):
        """Saves a COCO JSON with annotations compressed in RLE format and also saves corresponding image tiles.

        The COCO JSON is amended to add two keys for the full scene, referring to the folder name containing the
        photopea layers. This should correspond to the original Sentinel-1 VV geotiff filename so that the
        coordinates can be associated.

        Args:
            layer_pths (List[str]): List of path in a scene folder corresponding to Background.png, Layer 1.png, etc. Order matters.
            coco_output (dict): the dict defining the metadata and data container for the dataset that will be created
            coco_name (str, optional): the filename of the coco json. Defaults to "instances_slick_train_v2.json".

        Raises:
            ValueError: Errors if the path to the first file in layer_pths doesn't contain "Background"
            ValueError: Errors if a path to a label file in layer_pths doesn't contain "Layer"
        """

        # Make sure scene id is the same and we have reproj params
        assert scene_id == self.s1_scene_id, "First run  save_background_img_tiles!"
        assert self.s1_crs is not None, "First run  save_background_img_tiles!"
        assert self.s1_gcps is not None, "First run  save_background_img_tiles!"
        assert self.s1_image_shape is not None, "First run  save_background_img_tiles!"

        start_tile_n = self.global_tile_id
        for instance_path in layer_pths[1:]:
            # each label is of form class_instanceid.png
            if "_" not in str(instance_path):
                raise ValueError(f"The layer {instance_path} is not an instance label.")

            org_array = skio.imread(instance_path)
            if (
                len(org_array.shape) == 2 and "ambiguous" in instance_path
            ):  # hack to handle ambiguous images saved with vals 0 and 255 rather than correct color mapping
                org_array = org_array.clip(max=1)
                org_array = np.expand_dims(org_array, axis=2)
            with rasterio.open(instance_path) as src:
                profile = src.profile.copy()
                profile["driver"] = "GTiff"
                if org_array.shape[-1] == 1 and "ambiguous" in instance_path:
                    profile["count"] = 1
                else:
                    profile["count"] = 4
                profile["crs"] = self.s1_crs
                profile["gcps"] = self.s1_gcps

                with MemoryFile() as mem:
                    with mem.open(**profile) as m:
                        m.write(reshape_as_raster(org_array))
                        gcps_transform = transform.from_gcps(self.s1_gcps)
                        with WarpedVRT(
                            m,
                            src_crs=self.s1_crs,
                            src_transform=gcps_transform,
                            add_alpha=False,
                        ) as vrt_dst:
                            # arr is (c, h, w)
                            arr = vrt_dst.read(
                                out_shape=(vrt_dst.count, *self.s1_image_shape)
                            )

                            assert arr.shape[1:] == self.s1_image_shape

            tiled_arr = reshape_split(reshape_as_image(arr), (512, 512))
            # saving annotations
            tiles_n, _, _, _ = tiled_arr.shape
            for local_tile_id in range(tiles_n):
                instance_tile = tiled_arr[local_tile_id]
                big_image_fname = (
                    os.path.basename(os.path.dirname(instance_path)) + ".tif"
                )
                tile_fname = (
                    os.path.basename(os.path.dirname(instance_path))
                    + f"_vv-image_local_tile_{local_tile_id}.tif"
                )
                image_info = pycococreatortools.create_image_info(
                    self.global_tile_id, tile_fname, (512, 512)
                )
                image_info.update(
                    {
                        "big_image_id": self.big_image_id,
                        "big_image_original_fname": big_image_fname,
                    }
                )
                # go through each label image to extract annotation
                if image_info not in self.coco_output["images"]:
                    self.coco_output["images"].append(image_info)
                if instance_tile.shape[-1] == 1 and "ambiguous" in instance_path:
                    if 1 in np.unique(instance_tile):
                        class_id = 6
                        category_info = {
                            "id": class_id,
                            "is_crowd": True,
                        }  # forces compressed RLE format
                    else:
                        class_id = 0
                        category_info = {"id": class_id, "is_crowd": False}
                    binary_mask = instance_tile[:, :, -1]
                else:
                    class_id = get_layer_cls(
                        instance_tile, class_mapping_photopea, class_mapping_coco
                    )
                    if class_id != 0:
                        category_info = {
                            "id": class_id,
                            "is_crowd": True,
                        }  # forces compressed RLE format
                    else:
                        category_info = {"id": class_id, "is_crowd": False}
                    r, g, b = class_mapping_photopea[class_mapping_coco_inv[class_id]]
                    binary_mask = rgbalpha_to_binary(instance_tile, r, g, b).astype(
                        np.uint8
                    )

                annotation_info = pycococreatortools.create_annotation_info(
                    self.instance_id,
                    self.global_tile_id,
                    category_info,
                    binary_mask,
                    binary_mask.shape,
                    tolerance=0,
                )
                if annotation_info is not None:
                    annotation_info.update(
                        {
                            "big_image_id": self.big_image_id,
                            "big_image_original_fname": big_image_fname,
                        }
                    )
                    self.coco_output["annotations"].append(annotation_info)
                self.instance_id += 1
                self.global_tile_id += 1
            self.global_tile_id = start_tile_n
        self.big_image_id += 1
        self.global_tile_id = start_tile_n + tiles_n

    def create_coco_from_photopea_layers_no_tile(
        self, scene_id: str, layer_pths: List[str]
    ):
        """Saves a COCO JSON with annotations compressed in RLE format, without tiling and referring to the
            original Background.png images.

        The COCO JSON is amended to add two keys for the full scene, referring to the folder name containing the
        photopea layers. This should correspond to the original Sentinel-1 VV geotiff filename so that the
        coordinates can be associated.

        Args:
            layer_pths (list[str]): List of path in a scene folder corresponding to Background.png, Layer 1.png, etc. Order matters.
            coco_output (dict): the dict defining the metadata and data container for the dataset that will be created
            coco_name (str, optional): the filename of the coco json. Defaults to "instances_slick_train_v2.json".

        Raises:
            ValueError: Errors if the path to the first file in layer_pths doesn't contain "Background"
            ValueError: Errors if a path to a label file in layer_pths doesn't contain "Layer"
        """
        # Make sure scene id is the same and we have reproj params
        assert scene_id == self.s1_scene_id, "First run  save_background_img_tiles!"
        assert self.s1_crs is not None, "First run  save_background_img_tiles!"
        assert self.s1_gcps is not None, "First run  save_background_img_tiles!"
        assert self.s1_image_shape is not None, "First run  save_background_img_tiles!"

        for instance_path in layer_pths[1:]:
            # each label is of form class_instanceid.png
            if "_" not in str(instance_path):
                raise ValueError(f"The layer {instance_path} is not an instance label.")

            org_array = skio.imread(instance_path)
            with rasterio.open(instance_path) as src:
                profile = src.profile.copy()
                profile["driver"] = "GTiff"
                profile["count"] = 4
                profile["crs"] = self.s1_crs
                profile["gcps"] = self.s1_gcps

                with MemoryFile() as mem:
                    with mem.open(**profile) as m:
                        m.write(reshape_as_raster(org_array))
                        gcps_transform = transform.from_gcps(self.s1_gcps)
                        with WarpedVRT(
                            m,
                            src_crs=self.s1_crs,
                            src_transform=gcps_transform,
                            add_alpha=False,
                        ) as vrt_dst:
                            # arr is (c, h, w)
                            arr = vrt_dst.read(
                                out_shape=(vrt_dst.count, *self.s1_image_shape)
                            )
                            assert arr.shape[1:] == self.s1_image_shape

            big_image_original_fname = (
                os.path.basename(os.path.dirname(instance_path)) + ".tif"
            )
            big_image_fname = (
                os.path.basename(os.path.dirname(instance_path)) + "_Background.png"
            )
            image_info = pycococreatortools.create_image_info(
                self.big_image_id, big_image_fname, arr.shape
            )
            image_info.update(
                {
                    "big_image_id": self.big_image_id,
                    "big_image_original_fname": big_image_original_fname,
                }
            )
            # go through each label image to extract annotation
            if image_info not in self.coco_output["images"]:
                self.coco_output["images"].append(image_info)
            class_id = get_layer_cls(arr, class_mapping_photopea, class_mapping_coco)
            if class_id != 0:
                category_info = {
                    "id": class_id,
                    "is_crowd": True,
                }  # forces compressed RLE format
            else:
                category_info = {"id": class_id, "is_crowd": False}
            r, g, b = class_mapping_photopea[class_mapping_coco_inv[class_id]]
            binary_mask = rgbalpha_to_binary(arr, r, g, b).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                self.instance_id,
                self.big_image_id,
                category_info,
                binary_mask,
                binary_mask.shape,
                tolerance=0,
            )
            if annotation_info is not None:
                annotation_info.update(
                    {
                        "big_image_id": self.big_image_id,
                        "big_image_fname": big_image_fname,
                    }
                )
                self.coco_output["annotations"].append(annotation_info)
            self.instance_id += 1
        self.big_image_id += 1

    def save_coco_output(self, outpath: str = "./instances_slicks_test_v2.json"):
        # saving the coco dataset
        with open(f"{outpath}", "w") as output_json_file:
            json.dump(self.coco_output, output_json_file)

    def handle_aux_datasets(
        self, aux_datasets, scene_id, bounds, image_shape, **kwargs
    ):
        assert (
            len(aux_datasets) == 2 or len(aux_datasets) == 3
        )  # so save as png file need RGB or RGBA

        aux_dataset_channels = None
        for aux_ds in aux_datasets:
            if aux_ds == "ship_density":
                scene_date_month = get_scene_date_month(scene_id)
                ar = get_ship_density(bounds, image_shape, scene_date_month)
            else:
                ar = get_dist_array_from_vector(bounds, image_shape, aux_ds, **kwargs)

            ar = np.expand_dims(ar, 2)
            if aux_dataset_channels is None:
                aux_dataset_channels = ar
            else:
                aux_dataset_channels = np.concatenate(
                    [aux_dataset_channels, ar], axis=2
                )

        return aux_dataset_channels


def fetch_sentinel1_reprojection_parameters(
    scene_id: str, rescale: int = 8
) -> Tuple[List[float], Tuple[int, int], List[Any], Any]:
    src_path = f"s3://skytruth-cerulean-sa-east-1/outputs/rasters/{scene_id}.tiff"

    with rasterio.Env(AWS_REQUEST_PAYER="requester"):
        with rasterio.open(src_path) as src:
            gcps, crs = src.gcps
            gcps_transform = transform.from_gcps(gcps)

            with WarpedVRT(
                src,
                src_crs=crs,
                src_transform=gcps_transform,
                add_alpha=False,
            ) as vrt_dst:
                wgs84_bounds = vrt_dst.bounds
                vrt_width = vrt_dst.width
                vrt_height = vrt_dst.height

    return list(wgs84_bounds), (vrt_height, vrt_width), gcps, crs


def get_sentinel1_bounds(
    scene_id: str, url="https://nfwqxd6ia0.execute-api.eu-central-1.amazonaws.com"
) -> Tuple[float, float, float, float]:
    r = httpx.get(f"{url}/scenes/sentinel1/{scene_id}/info", timeout=None)
    try:
        r.raise_for_status()
        scene_info = r.json()
    except httpx.HTTPError:
        print(f"{scene_id} does not exist in TMS!")
        return None

    return tuple(scene_info["bounds"])  # type: ignore


def get_scene_date_month(scene_id: str) -> str:
    # i.e. S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5
    date_time_str = scene_id[17:32]
    date_time_obj = datetime.strptime(date_time_str, "%Y%m%dT%H%M%S")
    date_time_obj = date_time_obj.replace(day=1, hour=0, minute=0, second=0)
    return date_time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_dist_array_from_vector(
    bounds: Tuple[float, float, float, float],
    img_shape: Tuple[int, int, int],
    vector_ds: str,
    max_distance: int = 60000,
    aux_resample_ratio: int = 8,
):
    shp = fiona.open(vector_ds)
    resampled_shape = (
        img_shape[0] // aux_resample_ratio,
        img_shape[1] // aux_resample_ratio,
    )
    img_affine = rasterio.transform.from_bounds(
        *bounds, resampled_shape[0], resampled_shape[1]
    )
    rv_array, affine = dr.rasterize(
        shp,
        affine=img_affine,
        shape=resampled_shape,
    )

    if (rv_array == 0).all():
        dist_array = np.ones(img_shape) * 255

    else:
        my_dr = dr.DistanceRaster(
            rv_array,
            affine=affine,
        )
        dist_array = my_dr.dist_array

        # array values to match 0 - 255 where 255 is furthest away from feature
        dist_array = dist_array / (max_distance / 255)  # 60 km
        dist_array[dist_array >= 255] = 255

    # resample to original res
    upsampled_dist_array = skimage.transform.resize(dist_array, img_shape[0:2])
    upsampled_dist_array = upsampled_dist_array.astype(np.uint8)
    return upsampled_dist_array


def get_ship_density(
    bounds: Tuple[float, float, float, float],
    img_shape: Tuple[int, int],
    scene_date_month: str = "2020-08-01T00:00:00Z",
    max_dens=100,
    url="http://gmtds.maplarge.com/Api/ProcessDirect?",
) -> np.ndarray:
    h, w = img_shape
    bbox_wms = bounds[0], bounds[2], bounds[1], bounds[-1]
    query = {
        "action": "table/query",
        "query": {
            "engineVersion": 2,
            "sqlselect": [
                "category_column",
                "category",
                f"GridCrop(grid_float_4326, {', '.join([str(b) for b in bbox_wms])}) as grid_float",
            ],
            "table": {
                "query": {
                    "table": {"name": "ais/density"},
                    "where": [
                        [
                            {"col": "category_column", "test": "Equal", "value": "All"},
                            {"col": "category", "test": "Equal", "value": "All"},
                        ]
                    ],
                    "withgeo": True,
                }
            },
            "where": [
                [{"col": "time", "test": "Equal", "value": f"{scene_date_month}"}]
            ],
        },
    }

    qs = (
        f"request={json.dumps(query)}"
        "&uParams=action:table/query;formatType:tiff;withgeo:false;withGeoJson:false;includePolicies:true"
    )

    r = httpx.get(f"{url}{qs}", timeout=None, follow_redirects=True)
    try:
        r.raise_for_status()
        tempbuf = BytesIO(r.content)
        zipfile_ob = zipfile.ZipFile(tempbuf)
        cont = list(zipfile_ob.namelist())
        with rasterio.open(BytesIO(zipfile_ob.read(cont[0]))) as dataset:
            ar = dataset.read(
                out_shape=img_shape[0:2],
                out_dtype="uint8",
                resampling=Resampling.nearest,
            )
    except httpx.HTTPError:
        print("Failed to fetch ship density!")
        return None

    dens_array = ar / (max_dens / 255)
    dens_array[dens_array >= 255] = 255
    return np.squeeze(dens_array.astype("uint8"))
