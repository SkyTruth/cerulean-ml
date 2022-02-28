#!/usr/bin/env bash



function links_generator() {
    sceneIdid=$1
    # Get presign urls for 30 days
    days=604800
    
    export AWS_DEFAULT_REGION=eu-central-1
    raster_url=$(aws s3 presign s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff --expires-in $days)
    cog_detection_v1_url=$(aws s3 presign s3://skytruth-cerulean/outputs/rasterized_detection_v1/cog/$sceneIdid.tiff --expires-in $days)
    cog_infra_url=$(aws s3 presign s3://skytruth-cerulean/outputs/infrastructure/cog/$sceneIdid.tiff --expires-in $days) 

    echo $raster_url
    echo $cog_detection_v1_url
    echo $cog_infra_url

    query="raster_url=${raster_url}@cog_detection_v1_url=${cog_detection_v1_url}@cog_infra_url=${cog_infra_url}"
    # photopeaLink="https://skytruth.surge.sh/?$query"
    photopeaLink="http://localhost/?$query"
    echo $photopeaLink
}


# for list_file in grd_list/*; do
#     echo "$list_file "
#     while read sceneId; do
#         links_generator $sceneId
#     done <$list_file
# done

# test
links_generator S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3

