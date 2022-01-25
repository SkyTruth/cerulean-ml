#!/usr/bin/env bash

function rasterize() {
    sceneIdid=$1
    echo "$sceneIdid"
    [ ! -f data/rasters/$sceneIdid.tiff ] && aws s3 cp s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff data/rasters/
    [ ! -f data/vectors/$sceneIdid.geojson ] && aws s3 cp s3://skytruth-cerulean/outputs/vectors/$sceneIdid.geojson data/vectors/

    mkdir -p data/rasterized/
    mkdir -p data/cog/

    # convert to raster
    docker run --rm -v ${PWD}:/mnt/ cerulean/vec2ras:v1 \
        python rasterized.py \
        --vector_file=data/vectors/$sceneIdid.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/cog/$sceneIdid.tiff
    
    aws s3 cp data/cog/$sceneIdid.tiff s3://skytruth-cerulean/outputs/rasterized/
}

stringList=infrastructure.list,canonical_vessel.list,recent_vessel.list,old_vessel.list,natural_seep.list
for list_file in ${stringList//,/ }; do
    echo " $list_file "
    while read sceneId; do
        rasterize $sceneId
    done <data/$list_file
done
