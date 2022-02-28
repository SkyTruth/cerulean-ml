import click
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from rio_cogeo import cog_translate, cog_profiles
from rasterio.io import MemoryFile
from shapely import geometry

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
    
#     only_boundary(cog_file)




# def only_boundary(image):
#     # first, convert picture as RGBA
#     with Image.open(image).convert("RGBA") as img:
#         pixels = img.load()
#         for i in range(img.size[0]):
#             for j in range(img.size[1]):
#                 # if a pixel is not white...
#                 if pixels[i,j] != (255, 255, 255,255):
#                     #it becomes transparent
#                     pixels[i,j] = (0, 0, 0, 0)
#     # then the loops are over, we save
#     im = img.save(f"{image}.png")

    # with rasterio.open(cog_file) as infile:
    #     profile=infile.profile
    #     #
    #     # change the driver name from GTiff to PNG
    #     #
    #     profile['driver']='PNG'
    #     #
    #     # pathlib makes it easy to add a new suffix to a
    #     # filename
    #     #    
    #     png_filename=f"{cog_file}.png"
    #     raster=infile.read()
    #     with rasterio.open(png_filename, 'w', **profile) as dst:
    #         dst.write(raster)


