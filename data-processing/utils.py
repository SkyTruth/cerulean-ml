import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from rio_cogeo import cog_translate, cog_profiles
from rasterio.io import MemoryFile
import numpy as np
import os


def vector2raster(vector_file, raster_file, rasterize_file):
    """Convert vector files to raster, with the size of the raster file

    Args:
        vector_file (str): Location of a geojson file or shapefile
        raster_file (str): Location of a raster file
        rasterize_file (str): Location of rasterized output file

    """
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
    """Convert raster file(1 band) to COG file

    Args:
        rasterize_file (str): Location of raster file
        cog_file (str): Location of COG output file
        color (tuple): RGBA tuple color
    """
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
    """Convert raster file(1 band) to PNG file

    Args:
        rasterize_file (_type_): Location of raster file
        png_file (_type_): Location of PNG output file
        color (_type_): RGBA list color values
    """
    with rasterio.open(rasterize_file) as src_dst:
        arr = src_dst.read(1)
        heigth, width = arr.shape
        data = np.zeros((4, heigth, width)).astype(np.uint8)
        profile = dict(
            driver="PNG", width=width, height=heigth, count=4, dtype=data.dtype
        )
        # color = [255, 0, 255, 255]
        for index, item in enumerate(color):
            a = np.copy(arr)
            data[index] = np.where((a == 1), item, a)
        with rasterio.open(png_file, "w", **profile) as dst:
            dst.write(data)


def dir_(ouput_image):
    """Create folder if not exist for output file

    Args:
        ouput_image (str): Location of the file
    """
    dir_ = os.path.dirname(ouput_image)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
