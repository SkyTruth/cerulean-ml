#!/usr/bin/env bash

for list_file in grd_list/*
do
    echo "$list_file"
    while read sceneId; do
        # TODO, get presign urls
        raster_file="https://skytrue-test-remove.s3.amazonaws.com/rasters/$sceneId.tiff"
        overlay_raster_file="https://skytrue-test-remove.s3.amazonaws.com/rasterized/$sceneId.tiff"
        photopeaLink="https://skytruth.surge.sh/?raster_file=${overlay_raster_file}&raster_overlap=${overlay_raster_file}"
        echo $photopeaLink
    done <$list_file
done
