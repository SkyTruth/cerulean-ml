### Mounting
echo "Connection with GCS Fuse. accessing a path will dynamically mount it. any path to a bucket subdir can be accessed if the bucket is in the project"
mkdir -p data/
gcsfuse --implicit-dirs data/
