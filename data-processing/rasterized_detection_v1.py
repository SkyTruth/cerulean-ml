import click
from utils import vector2raster, raster2cog


@click.command(short_help="Script to convert vector to raster")
@click.option(
    "--vector_file",
    help="vector file",
    default="s3://skytruth-cerulean/outputs/vectors/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.geojson",
)
@click.option(
    "--raster_file",
    help="raster file",
    default="data/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.tiff",
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
    infra_color = (255, 255, 0, 1)
    raster2cog(output_rasterize_file, output_cog_file, infra_color)


if __name__ == "__main__":
    main()
