See the top level readme for instructions on using scripts necessary for setup, make_datasets.sh and move_datasets_to_ssd.sh.

## Run the Hydra Model Training CLI
TODO move to another section with description on not in sync with notebook.
Hydra is a robust configuration and experiment management tool. It is composed of a python module, `hydra`, and a `config` directory, with a hierarchy of yaml config files to define hyperparameters and settings for training your model.

First, install and activate the icevision environment (see above).

Then, run the script to move datasets to the gpu vm

```
bash scripts/move_datasets_to_ssd.sh
```

A set of experiments can be started by using Hydra's command line interface:

`python experiment.py --help`

and configs can be adapted on the fly like so:

```
python experiment.py model.pretrained=True datamodule.img_dir=/root/partitions/train-with-context-512/tiled_images/ datamodule.annotations_filepath=/root/partitions/train-with-context-512/instances_TiledCeruleanDatasetV2.json
```

or by editing the config file directly. Configs should specify good defaults (once you know what they are) and/or the set of configs necessary to reproduce an important experiment.

TODO instructions for viewing wandb logging and switching between fastai and icevision.
