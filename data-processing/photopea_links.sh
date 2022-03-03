#!/usr/bin/env bash

export AWS_DEFAULT_REGION=sa-east-1
export BUCKET=export BUCKET=skytruth-cerulean-sa-east-1

days_7=604800

function exist_object() {
    key=$1
    s3_obj="s3://${BUCKET}/${key}"
    existObj=$(aws s3 ls $s3_obj --summarize | grep "Total Objects: " | sed 's/[^0-9]*//g')
    if [ ! "$existObj" -eq "0" ]; then
        echo "https://${BUCKET}.s3.amazonaws.com/outputs/${key}"
    else
        echo "---"
    fi
}
function links_generator() {
    sceneIdid=$1
    # Get presign urls for 7 days
    # raster_url=$(aws s3 presign s3://${BUCKET}/outputs/rasters/$sceneIdid.tiff --expires-in $days_7)
    raster_url="https://skytruth-cerulean-sa-east-1.s3.amazonaws.com/outputs/rasters/$sceneIdid.tiff"
    png_detection_v1_url=$(exist_object outputs/rasterized_detection_v1/png/$sceneIdid.png)
    png_all_infrastructure_url=$(exist_object outputs/all_infrastructure/png/$sceneIdid.png)
    png_leaky_infrastructure_url=$(exist_object outputs/leaky_infrastructure/png/$sceneIdid.png)

    query="raster_url=${raster_url}@png_detection_v1_url=${png_detection_v1_url}@png_all_infra_url=${png_all_infrastructure_url}@png_leaky_infra_url=${png_leaky_infrastructure_url}"
    photopeaLink="https://skytruth.surge.sh/?$query"
    # photopeaLink="http://localhost/?$query"
    echo $photopeaLink
}

mkdir -p data/
links_file=data/photopea_link.csv
echo "type|sceneId|url" >$links_file

##### test
# links_generator S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3
# links_generator S1A_IW_GRDH_1SDV_20201005T025210_20201005T025235_034655_04093B_521A

for list_file in grd_list/*; do
    type_="$(basename -- $list_file)"
    type_="${type_%.*}"
    echo $type_
    while read sceneId; do
        echo $sceneId
        url=$(links_generator $sceneId)
        echo "$type_|$sceneId|=HYPERLINK(\"$url\",\"photopea\")" >>$links_file
    done <$list_file
done

aws s3 cp $links_file s3://${BUCKET}/outputs/photopea_links/
