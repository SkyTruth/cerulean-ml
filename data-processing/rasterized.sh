#!/usr/bin/env bash

function rasterize() {
    sceneIdid=$1
    echo "Processing... $sceneIdid"
    [ ! -f data/rasters/$sceneIdid.tiff ] && aws s3 cp s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff data/rasters/
    [ ! -f data/vectors/$sceneIdid.geojson ] && aws s3 cp s3://skytruth-cerulean/outputs/vectors/$sceneIdid.geojson data/vectors/

    ##################### rasterized detection v1 ####################
    mkdir -p data/rasterized_detection_v1/rasterized/
    mkdir -p data/rasterized_detection_v1/cog/

    python rasterized_detection_v1.py \
        --vector_file=data/vectors/$sceneIdid.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/rasterized_detection_v1/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/rasterized_detection_v1/cog/$sceneIdid.tiff

    aws s3 cp data/rasterized_detection_v1/cog/$sceneIdid.tiff s3://skytruth-cerulean/outputs/rasterized_detection_v1/cog/


    #################### rasterized Infraestructure file ####################
    mkdir -p data/infrastructure/rasterized
    mkdir -p data/infrastructure/cog
    mkdir -p data/infrastructure/clip_point

    python rasterized_infrastructure.py \
            --geojson_points_file="s3://bilge-dump-sample/Geotiffs/Global Coincident Infrastructure.geojson" \
            --raster_file=data/rasters/$sceneIdid.tiff \
            --output_rasterize_file=data/infrastructure/rasterized/$sceneIdid.tiff \
            --output_cog_file=data/infrastructure/cog/$sceneIdid.tiff \
            --output_clip_point_file=data/infrastructure/clip_point/$sceneIdid.geojson

    aws s3 cp data/infrastructure/cog/$sceneIdid.tiff s3://skytruth-cerulean/outputs/infrastructure/cog/
    aws s3 cp data/infrastructure/clip_point/$sceneIdid.geojson s3://skytruth-cerulean/outputs/infrastructure/clip_point/
}

# for list_file in grd_list/*; do
#     echo "$list_file "
#     while read sceneId; do
#         rasterize $sceneId
#     done <$list_file
# done

# test
rasterize S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3