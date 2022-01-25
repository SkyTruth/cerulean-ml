import click
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from rio_cogeo import cog_translate, cog_profiles
from rasterio.io import MemoryFile


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


def raster2cog(rasterize_file, cog_file):
    cmap_values = {0: (0, 0, 0, 0), 1: (182, 227, 196, 1)}
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


@click.command(short_help="Script to conveert vector to raster")
@click.option(
    "--vector_file",
    help="vector file",
    default="s3://skytruth-cerulean/outputs/vectors/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.geojson",
)
@click.option(
    "--raster_file",
    help="raster file",
    default="s3://skytruth-cerulean/outputs/rasters/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.tiff",
)
@click.option(
    "--output_rasterize_file",
    help="raster file",
    default="data/rasterized/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.tiff",
)
@click.option(
    "--output_cog_file",
    help="cog file",
    default="data/cog/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.tiff",
)
def main(vector_file, raster_file, output_rasterize_file, output_cog_file):
    vector2raster(vector_file, raster_file, output_rasterize_file)
    raster2cog(output_rasterize_file, output_cog_file)


if __name__ == "__main__":
    main()
