rm cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb # forgot to do this when making base image
### Mounting
echo "Connection with GCS Fuse. accessing a path will dynamically mount it. any path to a bucket subdir can be accessed if the bucket is in the project"
mkdir -p data/
gcsfuse --implicit-dirs cerulean data/
printf "alias cdata='gcsfuse --implicit-dirs cerulean data/'\n" >> /root/.bashrc #persists mounting
printf "alias jserve='jupyter lab --allow-root --no-browser'\n" >> /root/.bashrc #start jupyter

mkdir -p ~/work # same dir as remote_dir in makefile

# extra deps for coco creation, needs to be run after make start
# source .bashrc
# mamba env update --name fastai2 --file environment.yml --prune
# conda acticate fastai2
# pip install -e ceruleanml # install local ceruleanml package after deps installed with conda
