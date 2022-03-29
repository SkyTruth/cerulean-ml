import geopandas as gpd 
from shapely.geometry import box

whole_world = gpd.GeoDataFrame(index=[0], geometry=[box(-180.0000, -90.0000, 180.0000, 90.0000)], crs=4326)
distance = 0.5 # approx 55km
source = gpd.read_file("/Users/rodrigoalmeida/cerulean-ml/location_of_the_worlds_petroleum_fields__xtl.json")

buffered_geom = source.dissolve().buffer(0.5)
buffered_gpd = gpd.GeoDataFrame(geometry=buffered_geom)
inverted = buffered_gpd.overlay(whole_world, how="symmetric_difference")

inverted.to_file("inverted.json")