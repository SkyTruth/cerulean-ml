# Cerulean-ML Overview
This repo contains all of the assets needed to train, evaluate, and test  models befor ethey are put into staging or production on [cerulean-cloud](https://github.com/SkyTruth/cerulean-cloud).


# Local Setup, setting up `pre-commit`
This runs `black` (code formatter), `flake8` (linting), `isort` (import sorting) and `mypy` (type checks) every time you commit.

```
pip install pre-commit
pre-commit install
```

# TOC, folders

```
📦work
 ┣ 📂ceruleanml- the python module that contains a cli for dataset creation 
                and modules for dataset statistics, model training, model 
                evaluation, model output post processing and inference
 ┃ ┣ 📜README.md
 ┣ 📂data-processing- scripts for setting up the data annotation task and 
                      Photopea annotation UI
 ┃ ┣ 📂grd_list - lists of annotated grds used to load into the photopea UI
 ┃ ┣ 📂public- the location if the index.html with the annotation app. 
                  can be run as a static site on https://surge.sh or github 
                  pages or AWS. It's still live at https://skytruth.surge.
                  sh/ (a free devseed account).
 ┃ ┃ ┗ 📜index.html
 ┃ ┣ 📜README.md
 ┣ 📂gcpvm- terraform definition files that define the infra to deploy on 
            GCP. this includes a VM created from a base GCP VM image, an 
            attached SSD disk, and setup script to install custom 
            dependencies. Also includes a Makefile that defines workflow 
            commands for ssh and syncing files between local (for git 
            commiting) and remote VM (for work on files that need the 
            ceruleanml environments and/or a GPU)
 ┃ ┣ 📜README.md
 ┣ 📂notebooks- interactive notebooks for experimenting with the icevision 
            and fastai trainers, examining dataset statistics, querying 
            Sentinel-1 characteristics with eodag, exploring model results
 ┃ ┣ 📜README.md
 ┣ 📂scripts- contains scripts for VM setup, and convinience scripts for 
            icevision training and evaluation if you don't want to use the 
            notebooks. Also contains an example of using hydra to configure 
            an experiment.
 ┃ ┣ 📂config- the config folder for the hydra experiment.py
 ┃ ┣ 📜evaluation.py - runs evaluation to generate confusion matricies and 
                        metrics, saves plots.
 ┃ ┣ 📜experiment.py - example running simple icevision trainer with hydra. 
                        Not up to date with the notebook or ice_trianer.py
 ┃ ┣ 📜fastai2_unet_trainer_cv2-1channel-baseline.py - the fastai baseline 
                                that has good performance in terms of Dice, 
                                but issues with class mixing
 ┃ ┣ 📜fastaiunet_debug.py - fastai trainer, no hydra, for debugging in 
                            vscode
 ┃ ┣ 📜ice_trainer.py - icevision trainer, no hydra, for debugging in 
                            vscode
 ┃ ┣ 📜make_datasets.sh - run this when making datasets from scratch, 
                            uncomment which datasets you want to make
 ┃ ┗ 📜move_datasets_to_ssd.sh - move already created datasets from GCP to 
                                VM SSD with gsutil rsync
 ┣ 📂tests- tests for the dataset creation cli (most complex piece of the 
            ceruleanml package in terms of lines and code complexity)
 ┃ ┣ 📂fixtures- example data for the pytests
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜test_ceruleanml.py
 ┣ 📜.editorconfig
 ┣ 📜.gitignore
 ┣ 📜.isort.cfg
 ┣ 📜.pre-commit-config.yaml - no checks on files in scripts/, this is for  
                                checking the ceruleanml module
 ┣ 📜LICENSE
 ┣ 📜Makefile
 ┣ 📜README.md
 ┣ 📜environment.yml
 ┣ 📜requirements-test.txt
 ┣ 📜setup.cfg
 ┣ 📜setup.py
 ┗ 📜tox.ini
```
# VM Setup

After following the instructions in the [gcpvm README](gcpvm/README.md) to setup the VM and syncup the ceruleanml repo to the VM, we need to setup some python environments. our datasets, and start our development environment (Jupyter or VSCode).

We created three different python enviornments to handle different pieces of the Cerulean ML code base. As the project progressed fastai2 became used for less when we switched to focusing on developing tooling around the icevision mrcnn model.

### fastai2
This env is updated after vm creation with the root Makefile (not the Makefile in gcpvm). It's used for the fastai trainer. `make install` adds packages to the conda environment. See the Makefile for details.

### .ice-env
This environment is created with virtualenv. It's used to run the icevision trainer and do everything that requires icevision except for confusion matrix evaluation. It takes a few steps to create since we need to install a custom fork of icevision and activate the environment to run a step, which the Makefile can't handle.

Make sure to run on local prior to setting up the icevision env on vm.

```
# in local GCPVM directory:
make clone-ice
make syncup-ice
```

```
# in VM's work directory on:
make setup-icevision-env
source ./.ice-env/bin/activate # .ice-env is a hidden folder created in work/
make install-icevision-deps
cd ../icevision #this is cloned in the previous step from rbavery's fork
pip install -e .[dev]
cd ../work
pip install -e . # installing ceruleanml package and deps like pycocotools
```

### .ice-env-inf
This environment is created with virtualenv. It can likely be used for anything icevision related in this repo, but was created separately to test and develop evaluation with the icevision confusion matrix. It takes a few steps to create since we need to install a custom fork of icevision and activate the environment to run a step, which the Makefile can't handle.

```
make setup-icevision-inf-env
source ./.ice-env-inf/bin/activate # .ice-env is a hidden folder created in work/
make install-icevision-inf-deps
cd ../icevision #this is cloned in the previous step from rbavery's fork
pip install -e .[dev]
cd ../work
pip install -e . # installing ceruleanml package and deps like pycocotools
```

Whenever you need to activate the icevision environments, run 

```
source ./.ice-env/bin/activate
or
source ./.ice-env-inf/bin/activate
```

And these are available to the jupyter server as kernels.

Note: the gcpvm Makefile manages cloning the icevision repo locally, syncing it up, and syncing it down from the VM. this is because the icevision fork is another repo that may require the GPU and the icevision environments and it may be edited on the VM to provide new functionaility, like evaluating 6 class models with the SimpleConfusionMatrix.

You will also need to deactivate the icevision environments with `source deactivate` prior to activating the conda environments. 

# Other Setup

## Dataset Creation/Copying

To mount the buckets to the source imagery/labels exported from Photopea, `/root/data-cv2`, and the ceruleanml-biggpuregion bucket, `/root/data`, run

```
cdata
cdata2
```
From anywhere, but from `/root/work` is fine.

If a dataset has been created already and copied to GCP, the next step is to run `python scripts/move_datasets_to_ssd.sh`. This takes more than 8 minutes so its best to run it in `screen`. If you're not creating a fresh dataset, run move_datasets_to_ssd.sh after mounting since the premade CV2 dataset is already there. Make sure you have copied the dataset to the local SSD of the VM at /root. This will result in IO speed improvements. For example, parsing/loading the data with icevision from a GCP bucket takes a full 2 minutes compared to 17 seconds when data is already on the VM SSD.


If you want to make a new dataset with different parameters, you'll first need to authenticate with AWS on the VM. We use S1 tiles on AWS to georeference the Background.png images used to generate the Photopea labels.

Add the AWS CLI with authentication to be able to reach out to the Sentinel-1 PDS S3 bucket, to generate tiles:

```
apt install snapd
snap install aws-cli --classic
aws configure 
```
Add AWS Access Key ID and AWS Secret Access Key from AWS account, found locally at ~/.aws/credentials you only need to enter these fields leave the others blank.


`scripts/make_datasets.sh` is a good reference for what commands to run for different dataset creation options. Check it out to make sure you are creating the dataset you want. It is currently set up to transfer the dataset to gcp and delete the contents on the VM, since you would typically start a new GPU machine with less CPU cores for development and recopy the data back. 

Dataset creation is best run on a 32 core machine to make the train dataset in under 20 minutes, so see [gcpvm/terraform.tfvars](gcpvm/terraform.tfvars) for how to change the instance type. This requires having run the previous environment setup step for fastai, since we used that environment for dataset creation. Recommended to use a `screen` window for this.

After running `scripts/make_datasets.sh`, run `scripts/move_datasets_to_ssd.sh`

## Testing Dataset CLI
To install testing dependencies to run pytests for the dataset creation script, run this in the fastai environment

```
# For testing
pip install -r requirements_test.txt
```

## IDE Tips
To start a jupyter server, run
```
jserve
```
alternatively you can connect to the VM with VSCode remote explorer

```
Host icevision-Trainer
	HostName you-host-name-from-make-ssh
	IdentityFile /home/rave/cerulean-ml/gcpvm/.ssh/private_instance_gcp.pem
	User root
	LocalForward 8888 localhost:8888
	LocalForward 6006 localhost:6006
	IdentitiesOnly yes
```

A good way to get an understanding of how the dataset creation works is to use the VSCode debugger by setting breakpoints when running the tests in VSCode. You can pause and inspect variables, advance executions in a different ways, and traverse how the program runs, which can help to adapt the dataset creation to make it more flexible for different data types and image channel sources.

When using VSCode for debugging, it is recommende to set `justMyCode: False` if debugging other libraries (like icevision) and to also uncheck `Raised Exceptions` so that only errors stop the debugging process. Otherwise debugging can't progress.

You also need to set your interpreter to fastai2 when running the data.py pytests. To do so: CNTRL+SHFT+P to open the VSCode command palette, then select interpreter. You also need to install VSCode's python extension. 

autoDocstring is another good extension to have, which will create a formatted docstring templat eunder funcs whenever creating three quotes """.


