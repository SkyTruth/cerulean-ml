#!/usr/bin/env bash

function rasterize() {
    sceneIdid=$1
    mkdir -p data/rasterized/
    mkdir -p data/cog/
    
    # echo "$sceneIdid"
    [ ! -f data/rasters/$sceneIdid.tiff ] && aws s3 cp s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff data/rasters/
    [ ! -f data/vectors/$sceneIdid.geojson ] && aws s3 cp s3://skytruth-cerulean/outputs/vectors/$sceneIdid.geojson data/vectors/
    # convert to raster
    docker run --rm -v ${PWD}:/mnt/ cerulean/vec2ras:v1 \
        python rasterized.py \
        --vector_file=data/vectors/$sceneIdid.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/cog/$sceneIdid.tiff

    aws s3 cp data/cog/$sceneIdid.tiff s3://skytruth-cerulean/outputs/rasterized/
}

for list_file in grd_list/*; do
    echo "$list_file "
    while read sceneId; do
        rasterize $sceneId
    done <$list_file
done
