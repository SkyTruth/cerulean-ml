#!/usr/bin/env bash
export AWS_DEFAULT_REGION=eu-central-1
export BUCKET=skytruth-cerulean

function rasterize() {
    sceneIdid=$1
    echo "Processing... $sceneIdid"
    [ ! -f data/rasters/$sceneIdid.tiff ] && aws s3 cp s3://${BUCKET}/outputs/rasters/$sceneIdid.tiff data/rasters/
    [ ! -f data/vectors/$sceneIdid.geojson ] && aws s3 cp s3://${BUCKET}/outputs/vectors/$sceneIdid.geojson data/vectors/

    ##################### rasterized detection v1 ####################

    python rasterized_detection_v1.py \
        --vector_file=data/vectors/$sceneIdid.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/rasterized_detection_v1/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/rasterized_detection_v1/cog/$sceneIdid.tiff \
        --output_png_file=data/rasterized_detection_v1/png/$sceneIdid.png

    aws s3 cp data/rasterized_detection_v1/png/$sceneIdid.png s3://${BUCKET}/outputs/rasterized_detection_v1/png/

    #################### rasterized all Infraestructure file ####################

    [ ! -f data/detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.geojson ] && aws s3 cp s3://bilge-dump-sample/Geotiffs/detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.geojson data/
    python rasterized_points.py \
        --geojson_points_file=data/detections_2020_infrastructureOnly_v20210604_buffered_dissolved_centroids.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/all_infrastructure/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/all_infrastructure/cog/$sceneIdid.tiff \
        --output_clip_point_file=data/all_infrastructure/clip_point/$sceneIdid.geojson \
        --color="177,246,249,255" \
        --output_png_file=data/all_infrastructure/png/$sceneIdid.png


    aws s3 cp data/all_infrastructure/png/$sceneIdid.png s3://${BUCKET}/outputs/all_infrastructure/png/
    aws s3 cp data/all_infrastructure/clip_point/$sceneIdid.geojson s3://${BUCKET}/outputs/all_infrastructure/clip_point/

    #################### rasterized leaky infra ####################

    [ ! -f data/Global\ Coincident\ Infrastructure.geojson ] && aws s3 cp s3://bilge-dump-sample/Geotiffs/Global\ Coincident\ Infrastructure.geojson data/
    python rasterized_points.py \
        --geojson_points_file=data/Global\ Coincident\ Infrastructure.geojson \
        --raster_file=data/rasters/$sceneIdid.tiff \
        --output_rasterize_file=data/leaky_infrastructure/rasterized/$sceneIdid.tiff \
        --output_cog_file=data/leaky_infrastructure/cog/$sceneIdid.tiff \
        --output_clip_point_file=data/leaky_infrastructure/clip_point/$sceneIdid.geojson \
        --color="0,0,255,255" \
        --output_png_file=data/leaky_infrastructure/png/$sceneIdid.png

    aws s3 cp data/leaky_infrastructure/png/$sceneIdid.png s3://${BUCKET}/outputs/leaky_infrastructure/png/
    aws s3 cp data/leaky_infrastructure/clip_point/$sceneIdid.geojson s3://${BUCKET}/outputs/leaky_infrastructure/clip_point/
}

for list_file in grd_list/*; do
    echo "$list_file "
    while read sceneId; do
        rasterize $sceneId
    done <$list_file
done

# # # test
# rasterize S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3
# rasterize S1A_IW_GRDH_1SDV_20201005T025210_20201005T025235_034655_04093B_521A
