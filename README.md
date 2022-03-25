# cerulean-ml
Repo for Training ML assets for Cerulean


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

If there are new dependencies you find we need, you can add them in the environment.yaml and install with make install (the top level makefile not the terraform makefile).
