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
