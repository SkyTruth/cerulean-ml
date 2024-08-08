rm cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb # forgot to do this when making base image
### Mounting
echo "Connection with GCS Fuse. accessing a path will dynamically mount it. any path to a bucket subdir can be accessed if the bucket is in the project"
mkdir -p data/
mkdir -p data-cv2/
gcsfuse --implicit-dirs ceruleanml data/
gcsfuse --implicit-dirs cv2-training data-cv2/
printf "alias cdata='gcsfuse --implicit-dirs ceruleanml-biggpuregion /root/data/'\n" >> /root/.bashrc #persists mounting of processed data bucket
printf "alias cdata2='gcsfuse --implicit-dirs cv2-training /root/data-cv2/'\n" >> /root/.bashrc #persists mounting of source data bucket. treat as read only!
printf "alias jserve='jupyter lab --allow-root --no-browser'\n" >> /root/.bashrc #start jupyter
printf "export CUDA_HOME=/usr/local/cuda"
printf "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
printf "export PATH=$PATH:$CUDA_HOME/bin"
