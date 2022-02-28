SHELL = bash
# Settings
REMOTE_DIR=~/work
LOCAL_DIR=../

##### You should not have to touch stuff below this line
SHELL=bash
PATH_SSH_KEY=.ssh/private_instance_gcp.pem
PATH_VM_ID=.vm-id
PATH_VM_IP=.vm-ip
PATH_VM_NAME=.vm-name

VM_ID=$(shell cat ${PATH_VM_ID}) # GCP resource ID
VM_CONN=$(shell cat ${PATH_VM_IP}) # user@IP
VM_NAME=$(shell cat ${PATH_VM_NAME}) # name
VM_STR=$(addsuffix ${VM_NAME},name:)
VM_NAME_ZONE=$$(gcloud compute instances list --filter=$${VM_STR} --uri)

mkfile_path=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR=$(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

# sync code
syncup:
	ssh -i ${PATH_SSH_KEY} -o IdentitiesOnly=yes ${VM_CONN} "mkdir -p ${REMOTE_DIR}/"
	rsync -e "ssh -i ${PATH_SSH_KEY} -o IdentitiesOnly=yes" -avz --exclude="${LOCAL_DIR}.git/" --exclude-from=${LOCAL_DIR}.gitignore ${LOCAL_DIR} $(shell echo ${VM_CONN}:${REMOTE_DIR}/ | tr -d '[:space:]')
syncdown:
	rsync -e "ssh -i ${PATH_SSH_KEY} -o IdentitiesOnly=yes" -avz --exclude="${LOCAL_DIR}.git/" --exclude-from=${LOCAL_DIR}.gitignore $(shell echo ${VM_CONN}:${REMOTE_DIR}/ | tr -d '[:space:]') ${LOCAL_DIR}

# start/stop instance. the --filter step saves us from needing to provide the specific zone to the gcloud commands but requires
# different names for instances in different zones.
# not sure how else to do this since we only know the region ahead of time in variables.tf (no region arg for instances commands)
stop:
	gcloud compute instances stop $(VM_NAME_ZONE)
start:
	gcloud compute instances start $(VM_NAME_ZONE)
status:
	gcloud compute instances describe ${VM_NAME_ZONE}

# ssh into machine with jupyter lab
ssh:
	ssh -i ${PATH_SSH_KEY} -L 8888:localhost:8888 -o IdentitiesOnly=yes ${VM_CONN}