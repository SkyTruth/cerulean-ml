# cerulean-ml
Repo for Training ML assets for Cerulean

# Setup `pre-commit`
This runs `black` (code formatter), `flake8` (linting), `isort` (import sorting) and `mypy` (type checks) every time you commit.

```
pip install pre-commit
pre-commit install
```

# Install dependencies

```
pip install -e .
# For testing
pip install -r requirements_dev.txt
```

# Run the CLI
ceruleanml is both a python module and CLI for running data preprocessing dataset creation scripts. You can invoke it like so:

```
(fastai2) root@ml-jupyter:~/work# ceruleanml --help
Usage: ceruleanml [OPTIONS] COMMAND [ARGS]...

  CeruleanML CLI scripts for data processing.

Options:
  --help  Show this message and exit.

Commands:
  make-coco-dataset-no-context  Create the dataset with tiles but without...
  make-coco-dataset-no-tiles    Create the dataset without tiling and...
  make-coco-dataset-with-tiles  Create the dataset with tiles and context...
```

and to get more detailed help for a specific command (improving formatting of the help message is a TODO):

```
(fastai2) root@ml-jupyter:~/work# ceruleanml make-coco-dataset-no-context --help
Usage: ceruleanml make-coco-dataset-no-context [OPTIONS]

  Create the dataset with tiles but without context files (ship density and
  infra distance).

  Args:     class_folder_path (str): the path to the folder containing class
  folders ("Infrastructure", "Coincident", etc.)     coco_outdir (str): the
  path to save the coco json and the folder         of tiled images.

Options:
  --help  Show this message and exit.
```

# Setup

Deploy the VM, sync the git directory, and ssh with port forwarding
```
cd gcpvm
terraform init
terraform apply
make syncup
make ssh
```

Mount the gcp bucket, install custom dependencies in the fastai2 environment, and start a jupyter server.
```
cdata
cd work
make install
cd ..
jserve
```

Add the AWS CLI with authentication to be able to reach out to the Sentinel-1 PDS S3 bucket, to generate tiles:
```
apt install snapd
snap install aws-cli --classic
aws configure 
# Add AWS Access Key ID and AWS Secret Access Key from AWS account
```

If there are new dependencies you find we need, you can add them in the environment.yaml and install with make install (the top level makefile not the terraform makefile).
