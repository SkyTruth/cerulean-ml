#!/bin/bash
apt-get update
apt-get -y upgrade 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh
eval "$(miniconda3/bin/conda shell.bash hook)"
conda init
source .bashrc
conda install scikit-learn pandas jupyter ipython -y
conda install mamba -n base -c conda-forge -y

conda create -n fastai2 -y
conda activate fastai2
mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
mamba install -c fastchan fastai -y
mamba install -c conda-forge ipykernel ipywidgets -y
conda deactivate
conda install -n base -c conda-forge jupyterlab_widgets jupyterlab nb_conda_kernels -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
apt-get update
apt-get -y install cuda