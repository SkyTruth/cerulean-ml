instance-type = "n1-standard-8" # "n1-standard-8" is a good default. only n1 types can add gpus https://stackoverflow.com/questions/53968149/add-gpu-to-an-existing-vm-instance-google-compute-engine
gpu-count     = 1
gpu-type      = "nvidia-tesla-t4"
location      = "europe-west4"
zone          = "europe-west4-a"
