rm cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb # forgot to do this when making base image
### Mounting
echo "Connection with GCS Fuse. accessing a path will dynamically mount it. any path to a bucket subdir can be accessed if the bucket is in the project"
mkdir -p data/
gcsfuse --implicit-dirs cerulean data/
printf "alias cdata='gcsfuse --implicit-dirs cerulean data/'\n" >> /root/.bashrc #persists mounting
printf "alias jserve='jupyter lab --allow-root --no-browser'\n" >> /root/.bashrc #start jupyter

# extra deps for coco creation, needs to be run after make start
# mamba activate fastai2
# mamba install cython -y
# pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# pip install
# pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
# mamba deactivate
