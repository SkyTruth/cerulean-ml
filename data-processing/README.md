# Script to proceess data for labeling

This section contains tools to develop the training dataset. 

## Vector to raster

Script to convert vector data into raster - COG(cloud optimized geotiff)

```sh
    export AWS_ACCESS_KEY_ID=xyz..
    export AWS_SECRET_ACCESS_KEY=xyz..
    docker-compose build
    ./rasterized.sh
```

## Generate Links to open Photopea

This creates links to a photopea interface populated with the matching Sentinel-1 scene for each link. The interface is customized for the 6 class annotation task.


```sh

./photopea_links.sh

```

## Site to load raster files

```sh

surge public/ skytruth.surge.sh

```

## TODO confirm where this interface lives add notes/guide on deploying in skytruth aws environment add callout

