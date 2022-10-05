import glob
import os
import random
import subprocess

source_path = "./data-cv2/"
dest_path = "./data/partitions/"

classes = ["Coincident", "Infrastructure", "Old", "Recent", "Natural_Seep", "Ambiguous"]


def get_scenes(path):
    """Takes a path where annotations are written to. Returns the scene folders within that path.
    Args:
        path (str): the path where annotations are written to.
    Returns:
        scenes (list): list of annotated S1 scenes.
    """
    scenes = [(x) for x in list(os.scandir(path)) if x.is_dir()]
    scenes = [(x.name) for x in scenes]
    return scenes

def partition_scenes(c, train_frac, val_frac, test_frac):
    """Applies random selection of scenes for mutually exclusive partitions (train, validation, test). 
    Partition fractions must range from 0.0 to 1.0.
    Args:
        c (str): a target class.
        train_frac (float): Percent of items to allocate to training.
        val_frac (float): Percent of items to allocate to validation. 
        test_frac (float): Percent of items to allocate to test. 
    Returns:
        train_scenes (list): List of train items.
        val_scenes_select (list): List of validation items.
        test_scenes_select (list): List of test items.
    """
    print("Selecting from class: ", c)
    scenes = get_scenes(f"{source_path}/{c}/*")
    random.seed(4)
    random.shuffle(scenes)
    len_train = int(len(scenes)*train_frac)
    len_val = int(len(scenes)*(val_frac))
    len_test = len(scenes)-(len_train+len_val)
    # assert test_frac == (len_test/len(scenes))

    train_scenes = scenes[0:len_train]
    val_scenes = scenes[len_train:len_train+len_val]
    test_scenes = scenes[len_train+len_val:]

    return train_scenes, val_scenes, test_scenes


for c in classes:
    train_scenes, val_scenes, test_scenes = partition_scenes(c)

with open(os.path.join(dest_path, "train_scenes.txt"), "w") as f:
    for item in train_scenes:
        f.write("%s\n" % item)

with open(os.path.join(dest_path, "val_scenes.txt"), "w") as f:
    for item in val_scenes:
        f.write("%s\n" % item)

with open(os.path.join(dest_path, "test_scenes.txt"), "w") as f:
    for item in test_scenes:
        f.write("%s\n" % item)
