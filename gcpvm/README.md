# Deploy GPU ML instance in GCP

## Setup

Follow the steps described in [this](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/provider_reference#authentication) to setup GCP authentication.

Install the [terraform CLI](https://learn.hashicorp.com/tutorials/terraform/install-cli).

## Import the base image

In case your project doesn't have a the `ubuntu-2004-cuda113-fastai-cerulean` you can create it from the `ubuntu-os-cloud/ubuntu-2004-focal-v20220110` image using a vm deployed with this image. `instance.tf` will need to use the `ubuntu-os-cloud/ubuntu-2004-focal-v20220110` image (see the comments in that file) and then the `ubuntu-2004-cuda113-fastai-cerulean` image can be created from the stopped instance with:

```
gcloud compute images create ubuntu-2004-cuda113-fastai-cerulean --project=cerulean-338116 --description=An\ image\ created\ from\ the\ cerulean-ml\ startup-script.sh\ in\ gcpvm/$'\n'$'\n'This\ creates\ an\ image\ with\ conda,\ mamba,\ docker,\ environments\ with\ fastai,\ icevision,\ git,\ jupyter,\ dynamic\ mounting\ of\ gcp\ buckets\ and\ other\ commonly\ used\ dev\ tools --family=ubuntu-2004-cuda113-fastai-cerulean --source-disk=ml-jupyter-ad7ada77-2be7-d3a3-62a8-abe8015e64f6-jupyter-disk --source-disk-zone=europe-west1-b --storage-location=europe-west1
```

## Adapt `variables.tf` file

Navigate to the folder containing `main.tf`. Adapt the `variables.tf` file as needed, specifically the `project`, the `instance-type` and the `location`. Currently we are using european regions since the Sentinel-1 data source is in Frankfurt.

## Deploy

Navigate to the folder containing the `main.tf` file. Run `terraform init`.

Check your deployment with `terraform plan`.

If you get a credentials error, you might need to run `gcloud auth application-default login`.

You can create your instance with `terraform apply`.

This will create a GCP Compute instance, and save in your local machine a private ssh key (in `.ssh/`), and a series of `.vm-X` files containing identity information for your instance. **Do not delete or modify this files!**

## The VM

This VM contains a conda environment `fastai2` that can be activated with `mamba activate fastai2` and edited by editing `minimal-start-up-script.sh`, which runs commands when `terraform apply` is called. See the `start-up-script.sh` for reference (this was used to create the custom cerulean base image) and in particular use `-y` flags when installing packages with `mamba` so that there is no waiting for manual response.

The VM also comes with docker and jupyter with port forwarding to your local machine (you can copy and paste a jupyter link on the VM to your local machine's browser).

## `make` tools

You can now use the set of tools included in the `Makefile`. Adapt this file if needed in case you want to change the remote and local path to copy files into the instance.

- `make ssh`: connects to your instance in your shell. This also maps the port 8888 in the instance to your localhost, allowing you to serve a jupyter instance via this ssh tunnel (for instance by running `jupyter lab --allow-root`).
- `make start`, `make stop`, `make status` : Start, stop and check the status of your instance. **Important: if you are not using your instance, make sure to run `make stop` to avoid excessive costs! Don't worry, your instance state and files are safe.**
- `make syncup` and `make syncdown`: Copies files in your folder from and to your instance.

## Destroy

When you finish all work associated with this instance make sure to run `terraform destroy`. This will delete the ssh key in `.ssh` and all `.vm-X` files.

**Important: when you destroy your instance, all files and instance state are deleted with it so make sure to back them up to GCS or locally if needed!**


## Notes on the Instance

The instance will have access to all buckets in the project. These buckets will be mounted under the directory `/root/data`. They can only be accessed by specifying paths to the contents of their subdirs. See the [gcsfuse mounting instructions](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/mounting.md) for more details.
