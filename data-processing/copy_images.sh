#!/usr/bin/env bash

function rasterize() {
    sceneIdid=$1
    aws s3 cp s3://skytruth-cerulean/outputs/rasters/$sceneIdid.tiff s3://skytruth-cerulean-sa-east-1/outputs/rasters/$sceneIdid.tiff
}

for list_file in grd_list/*; do
    echo "$list_file "
    while read sceneId; do
        rasterize $sceneId
    done <$list_file
done

# # # test
# rasterize S1A_IW_GRDH_1SDV_20200804T045214_20200804T045239_033752_03E97F_88D3
