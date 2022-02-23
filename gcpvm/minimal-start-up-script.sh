#!/bin/bash
# Add additional installs or env setups if needed
# GFUSE Setup, this can be deleted once the base image is updated
apt-get update && apt-get install -y lsb-release && apt-get clean all
apt-get -y install curl
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y gcsfuse
apt-get install -y apt-transport-https ca-certificates gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install -y google-cloud-sdk

### Mounting
echo "Connection with GCS Fuse. accessing a path will dynamically mount it. any path to a bucket subdir can be accessed if the bucket is in the project"
mkdir data/
gcsfuse --implicit-dirs data/
