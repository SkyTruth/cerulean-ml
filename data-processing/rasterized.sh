#!/usr/bin/env bash
export AWS_DEFAULT_REGION=eu-central-1
export BUCKET=skytruth-cerulean

function rasterize() {
    sceneIdid=$1
    echo "Processing... $sceneIdid"
    [ ! -f data/rasters/$sceneIdid.tiff ] && aws s3 cp s3://${BUCKET}/outputs/rasters/$sceneIdid.tiff data/rasters/
    [ ! -f data/vectors/$sceneIdid.geojson ] && aws s3 cp s3://${BUCKET}/outputs/vectors/$sceneIdid.geojson data/vectors/

    ##################### rasterized detection v1 ####################
    mkdir -p data/rasterized_detection_v1/rasterized/
    mkdir -p data/rasterized_detection_v1/cog/

    python rasterized_detection_v1.py \
        --vector_file=data/vectors/$sceneIdid.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/rasterized_detection_v1/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/rasterized_detection_v1/cog/$sceneIdid.tiff

    aws s3 cp data/rasterized_detection_v1/cog/$sceneIdid.tiff s3://${BUCKET}/outputs/rasterized_detection_v1/cog/

    #################### rasterized all Infraestructure file ####################
    mkdir -p data/all_infrastructure/rasterized
    mkdir -p data/all_infrastructure/cog
    mkdir -p data/all_infrastructure/clip_point

    [ ! -f data/detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.geojson ] && aws s3 cp s3://bilge-dump-sample/Geotiffs/detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.geojson data/
    python rasterized_points.py \
        --geojson_points_file=data/detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/all_infrastructure/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/all_infrastructure/cog/$sceneIdid.tiff \
        --output_clip_point_file=data/all_infrastructure/clip_point/$sceneIdid.geojson \
        --color="177,246,249,1"

    aws s3 cp data/all_infrastructure/cog/$sceneIdid.tiff s3://${BUCKET}/outputs/all_infrastructure/cog/
    aws s3 cp data/all_infrastructure/clip_point/$sceneIdid.geojson s3://${BUCKET}/outputs/all_infrastructure/clip_point/

    #################### rasterized leaky infra ####################
    mkdir -p data/leaky_infrastructure/rasterized
    mkdir -p data/leaky_infrastructure/cog
    mkdir -p data/leaky_infrastructure/clip_point

    [ ! -f data/Global\ Coincident\ Infrastructure.geojson ] && aws s3 cp s3://bilge-dump-sample/Geotiffs/Global\ Coincident\ Infrastructure.geojson data/
    python rasterized_points.py \
        --geojson_points_file=data/Global\ Coincident\ Infrastructure.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/leaky_infrastructure/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/leaky_infrastructure/cog/$sceneIdid.tiff \
        --output_clip_point_file=data/leaky_infrastructure/clip_point/$sceneIdid.geojson \
        --color="0,0,255,1"

    aws s3 cp data/leaky_infrastructure/cog/$sceneIdid.tiff s3://${BUCKET}/outputs/leaky_infrastructure/cog/
    aws s3 cp data/leaky_infrastructure/clip_point/$sceneIdid.geojson s3://${BUCKET}/outputs/leaky_infrastructure/clip_point/
}

for list_file in grd_list/*; do
    echo "$list_file "
    while read sceneId; do
        rasterize $sceneId
    done <$list_file
done

# # test
# rasterize S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3
