#!/usr/bin/env bash

stringList=infrastructure.list,canonical_vessel.list,recent_vessel.list,old_vessel.list,natural_seep.list
for list_file in ${stringList//,/ }
do
    echo "$list_file"
    while read sceneId; do
        photopeaLink=$(node photopea_links.js "https://skytrue-test-remove.s3.amazonaws.com/rasters/$sceneId.tiff,https://skytrue-test-remove.s3.amazonaws.com/cog/$sceneId.tiff")
        echo $photopeaLink
    done <data/$list_file
done
