import click
import rasterio
import geopandas as gpd
from shapely import geometry
from utils import vector2raster, raster2cog


def raster_bounds_poly(raster_file):
    # Raster bounds
    srcDataset = rasterio.open(raster_file)
    gcps, crs = srcDataset.get_gcps()
    poly = geometry.Polygon([[gcp.x, gcp.y] for gcp in gcps])
    polygon = geometry.box(*poly.bounds, ccw=True)
    return polygon


def clipArea(geojson_points_file, raster_bbox_poly, output_clip_point_file):
    geodf = gpd.read_file(geojson_points_file)
    points = gpd.clip(geodf, raster_bbox_poly)
    polygons = points.buffer(0.009, cap_style=3)
    polygons.to_file(output_clip_point_file, driver="GeoJSON")
    # return polygons


@click.command(short_help="Script to conveert vector to raster")
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
def main(
    geojson_points_file, raster_file, output_rasterize_file, output_cog_file, output_clip_point_file
):
    raster_bbox_poly = raster_bounds_poly(raster_file)
    clipArea(geojson_points_file, raster_bbox_poly, output_clip_point_file)
    vector2raster(output_clip_point_file, raster_file, output_rasterize_file)

    infra_color = (0, 0, 255, 1)
    raster2cog(output_rasterize_file, output_cog_file, infra_color)


if __name__ == "__main__":
    main()
