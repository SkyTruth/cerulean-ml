#!/usr/bin/env bash

export AWS_DEFAULT_REGION=eu-central-1
export BUCKET=skytruth-cerulean
days_7=604800

function exist_object() {
    s3_obj=$1
    existObj=$(aws s3 ls $s3_obj --summarize | grep "Total Objects: " | sed 's/[^0-9]*//g')
    if [ ! "$existObj" -eq "0" ]; then
        obj_url=$(aws s3 presign $s3_obj --expires-in $days_7)
        echo $obj_url
    else
        echo "---"
    fi
}
function links_generator() {
    sceneIdid=$1
    # Get presign urls for 7 days
    raster_url=$(aws s3 presign s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff --expires-in $days_7)
    png_detection_v1_url=$(exist_object s3://skytruth-cerulean/outputs/rasterized_detection_v1/png/$sceneIdid.png)
    png_all_infrastructure_url=$(exist_object s3://skytruth-cerulean/outputs/all_infrastructure/png/$sceneIdid.png)
    png_leaky_infrastructure_url=$(exist_object s3://skytruth-cerulean/outputs/leaky_infrastructure/png/$sceneIdid.png)

    query="raster_url=${raster_url}@png_detection_v1_url=${png_detection_v1_url}@png_all_infra_url=${png_all_infrastructure_url}@png_leaky_infra_url=${png_leaky_infrastructure_url}"
    photopeaLink="https://skytruth.surge.sh/?$query"
    # photopeaLink="http://localhost/?$query"
    echo $photopeaLink
}

mkdir -p data/
echo "type|sceneId|url" >data/photopea_Link.csv

##### test
# links_generator S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3
# links_generator S1A_IW_GRDH_1SDV_20201005T025210_20201005T025235_034655_04093B_521A

for list_file in grd_list/*; do
    type_="$(basename -- $list_file)"
    type_="${type_%.*}"
    echo $type_
    while read sceneId; do
        # links_generator $sceneId
        url=$(links_generator $sceneId)
        echo "$type_|$sceneId|=HYPERLINK(\"$url\",\"photopea\")" >>data/photopea_Link.csv
    done <$list_file
done

aws s3 cp data/photopea_link.csv s3://skytruth-cerulean/outputs/photopea_links/
