import click
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from rio_cogeo import cog_translate, cog_profiles
from rasterio.io import MemoryFile
from shapely import geometry
import numpy as np
import os


def vector2raster(vector_file, raster_file, rasterize_file):
    geodf = gpd.read_file(vector_file)
    geodf = geodf.explode()
    with rasterio.open(raster_file) as srcDataset:
        # Get values for destination file
        kwargs = srcDataset.profile.copy()
        rio_gcps, crs = srcDataset.get_gcps()

        approx_transform = rasterio.transform.from_gcps(rio_gcps)
        kwargs.update({"transform": approx_transform, "crs": crs})

        dst_img = rasterize(
            [(shape, 1) for shape in geodf["geometry"]],
            out_shape=srcDataset.shape,
            transform=approx_transform,
            fill=0,
            all_touched=True,
            dtype=rasterio.uint8,
        )
        # Write raster
        with rasterio.open(rasterize_file, "w", **kwargs) as dst:
            dst.write(dst_img, indexes=1)


def raster2cog(rasterize_file, cog_file, color):
    cmap_values = {0: (0, 0, 0, 0), 1: color}
    with rasterio.open(rasterize_file) as src_dst:
        profile = src_dst.profile.copy()
        profile["nodata"] = 0
        arr = src_dst.read()
        with MemoryFile() as memfile:
            with memfile.open(**profile) as mem:
                mem.write(arr)
                mem.write_colormap(1, cmap_values)
                dst_profile = cog_profiles.get("deflate")
                cog_translate(
                    mem,
                    cog_file,
                    dst_profile,
                    in_memory=True,
                    allow_intermediate_compression=True,
                    quiet=False,
                )


def raster2png(rasterize_file, png_file, color):
    with rasterio.open(rasterize_file) as src_dst:
        arr = src_dst.read(1)
        heigth, width = arr.shape
        data = np.zeros((4, heigth, width)).astype(np.uint8)
        profile = dict(driver="PNG", width=width, height=heigth, count=4, dtype=data.dtype)
        # color = [255, 0, 255, 255]
        for index, item in enumerate(color):
            a = np.copy(arr)
            data[index] = np.where((a == 1), item, a)
        with rasterio.open(png_file, "w", **profile) as dst:
            dst.write(data)


def dir_(ouput_image):
    dir_ = os.path.dirname(ouput_image)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
