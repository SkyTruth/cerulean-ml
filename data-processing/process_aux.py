import geopandas as gpd
import rasterio
from shapely.geometry import box


def invert_geom(
    ds="location_of_the_worlds_petroleum_fields__xtl.json", out="inverted.json"
):
    whole_world = gpd.GeoDataFrame(
        index=[0], geometry=[box(-180.0000, -90.0000, 180.0000, 90.0000)], crs=4326
    )
    distance = 0.5  # approx 55km
    source = gpd.read_file()

    buffered_geom = source.dissolve().buffer(0.5)
    buffered_gpd = gpd.GeoDataFrame(geometry=buffered_geom)
    inverted = buffered_gpd.overlay(whole_world, how="symmetric_difference")

    inverted.to_file(out)


def process_csv_infra(
    ds="ml_aux_files_detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.csv",
    out="infra_locations.json",
):
    df_nongeo = gpd.read_file(ds)
    df_geo = gpd.GeoDataFrame(df_nongeo, geometry=gpd.points_from_xy(df.Lon, df.Lat))
    df_geo.to_file(out)


def process_tif_ship_density(f="ShipDensity_Commercial1.tif", out="ship_density.json"):
    with rasterio.open(f) as src:
        density = src.read(0)
