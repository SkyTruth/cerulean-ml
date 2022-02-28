#!/usr/bin/env bash

export AWS_DEFAULT_REGION=eu-central-1
export BUCKET=skytruth-cerulean

function links_generator() {
    sceneIdid=$1
    # Get presign urls for 7 days
    days_7=604800

    raster_url=$(aws s3 presign s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff --expires-in $days_7)
    cog_detection_v1_url=$(aws s3 presign s3://skytruth-cerulean/outputs/rasterized_detection_v1/cog/$sceneIdid.tiff --expires-in $days_7)
    cog_all_infrastructure_url=$(aws s3 presign s3://skytruth-cerulean/outputs/all_infrastructure/cog/$sceneIdid.tiff --expires-in $days_7)
    cog_leaky_infrastructure_url=$(aws s3 presign s3://skytruth-cerulean/outputs/leaky_infrastructure/cog/$sceneIdid.tiff --expires-in $days_7)

    # echo $raster_url
    # echo $cog_detection_v1_url
    # echo $cog_all_infrastructure_url
    # echo $cog_leaky_infrastructure_url

    query="raster_url=${raster_url}@cog_detection_v1_url=${cog_detection_v1_url}@cog_all_infra_url=${cog_all_infrastructure_url}@cog_leaky_infra_url=${cog_leaky_infrastructure_url}"
    photopeaLink="https://skytruth.surge.sh/?$query"
    # photopeaLink="http://localhost/?$query"
    echo $photopeaLink
}

mkdir -p data/
echo "type|sceneId|url"> data/photopea_Link.csv

# test
# links_generator S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3

for list_file in grd_list/*; do
    type_="$(basename -- $list_file)"
    type_="${type_%.*}"
    echo $type_
    while read sceneId; do
        url=$(links_generator $sceneId)
        echo "$type_|$sceneId|=HYPERLINK(\"$url\",\"photopea\")" >>data/photopea_Link.csv
    done <$list_file
done

aws s3 cp data/photopea_link.csv s3://skytruth-cerulean/outputs/photopea_links/

