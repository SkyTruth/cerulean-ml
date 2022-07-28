instance-type = "n1-standard-8" # "n1-standard-8" is a good default. only n1 types can add gpus https://stackoverflow.com/questions/53968149/add-gpu-to-an-existing-vm-instance-google-compute-engine
gpu-count     = 1
gpu-type      = "nvidia-tesla-t4" #TODO alternative GPU options for europe-west4
location      = "europe-west4" # don't change this unless you knbow you need to. only west4 has a wide range of GPU types
zone          = "europe-west4-a"
