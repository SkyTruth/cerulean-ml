import click
import geopandas as gpd
import rasterio
from shapely import geometry
from utils import dir_, raster2png, vector2raster


def raster_bounds_poly(raster_file):
    """Get bounds for for rater file.
    Note:
        This function works for files that has gcps

    Args:
        raster_file (str): Location of raster file

    Returns:
        geometry: bound as polygon
    """
    # Raster bounds
    srcDataset = rasterio.open(raster_file)
    gcps, crs = srcDataset.get_gcps()
    poly = geometry.Polygon([[gcp.x, gcp.y] for gcp in gcps])
    polygon = geometry.box(*poly.bounds, ccw=True)
    return polygon


def clipArea(geojson_points_file, raster_bbox_poly, output_clip_point_file):
    """Clip vector points files for a given bbox

    Args:
        geojson_points_file (str): Location of geojson points files
        raster_bbox_poly (geometry): Bbox geometry of a reaster files
        output_clip_point_file (str): Location of ouput clipped geojson file
    """
    geodf = gpd.read_file(geojson_points_file)
    points = gpd.clip(geodf, raster_bbox_poly)
    polygons = points.buffer(0.005)
    polygons.to_file(output_clip_point_file, driver="GeoJSON")
    # return polygons


@click.command(short_help="Script to convert vector(points) to raster")
@click.option(
    "--geojson_points_file",
    help="all infra geojson fille",
    default="s3://bilge-dump-sample/Geotiffs/Global Coincident Infrastructure.geojson",
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
@click.option(
    "--output_clip_point_file",
    help="cog file",
    default="data/clip_point/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.geojson",
)
@click.option(
    "--color",
    help="color to rasterize",
    default="0,0,255,1",
)
@click.option(
    "--output_png_file",
    help="PNG file",
    default="data/png/S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3.png",
)
def main(
    geojson_points_file,
    raster_file,
    output_rasterize_file,
    output_cog_file,
    output_clip_point_file,
    color,
    output_png_file,
):
    dir_(output_rasterize_file)
    dir_(output_cog_file)
    dir_(output_clip_point_file)
    dir_(output_png_file)
    raster_bbox_poly = raster_bounds_poly(raster_file)
    clipArea(geojson_points_file, raster_bbox_poly, output_clip_point_file)
    vector2raster(output_clip_point_file, raster_file, output_rasterize_file)
    t_color = [int(i) for i in color.split(",")]
    # raster2cog(output_rasterize_file, output_cog_file, t_color)
    raster2png(output_rasterize_file, output_png_file, t_color)


if __name__ == "__main__":
    main()
