```
📦ceruleanml
 ┣ 📜__init__.py
 ┣ 📜cli.py - the dataset creation CLI
 ┣ 📜coco_load_fastai.py - a fastai data loader created from an icevision 
                            record collection created from a COCO JSON and 
                            folder of image tiles (the output of cli.py)
 ┣ 📜coco_stats.py - module for calculating statistics with scikit image 
                        region props
 ┣ 📜data.py - dataset creation module used by cli.py
 ┣ 📜evaluation.py - computes confusion matrices, normalized by predictions 
                        and groundtruth and in count units on instance-basis. used in evaluation notebook.
 ┣ 📜inference.py - funcs for saving models in different formats and post 
                        processing model results.
 ┣ 📜load_negative_tiles.py - negative tile parser, used in preprocess.py
 ┣ 📜preprocess.py - loads positive samples with a custom COCO parser subclassed from icevision and applies optional preprocessing to the dataset, removing small area samples.
 ┗ 📜random_scene_select.py - script used to generate partitions/ folder containing train, val, and test set images
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
