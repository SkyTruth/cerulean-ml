import os
import random

source_path = "./data-cv2/"
dest_path = "./data/partitions/"

classes = ["Coincident", "Infrastructure", "Old", "Recent", "Natural_Seep", "Ambiguous"]

train_frac, val_frac = 0.7, 0.2

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

def partition_scenes(c, train_frac, val_frac):
    """Applies random selection of scenes for mutually exclusive partitions (train, validation, test). 
    Partition fractions must range from 0.0 to 1.0.
    Args:
        c (str): a target class.
        train_frac (float): Percent of items to allocate to training.
        val_frac (float): Percent of items to allocate to validation. 
    Returns:
        train_scenes (list): List of train items.
        val_scenes (list): List of validation items.
        test_scenes (list): List of test items.
    """
    print("Selecting from class: ", c)
    scenes = get_scenes(f"{source_path}/{c}/*")
    random.seed(4)
    random.shuffle(scenes)
    len_train = int(len(scenes)*train_frac)
    len_val = int(len(scenes)*(val_frac))
    len_test = len(scenes)-(len_train+len_val)

    train_scenes = scenes[0:len_train]
    val_scenes = scenes[len_train:len_train+len_val]
    test_scenes = scenes[len_train+len_val:]
    
    # Check that test scenes is not empty
    assert len(test_scenes) > 0
    
    # Check for mutual exclusivity
    assert len(list(set(train_scenes) ^ set(val_scenes)))  == len_train + len_val
    assert len(list(set(val_scenes) ^ set(test_scenes)))  == len_val + len_test
    assert len(list(set(train_scenes) ^ set(test_scenes)))  == len_train + len_test

    return train_scenes, val_scenes, test_scenes


for c in classes:
    train_scenes, val_scenes, test_scenes = partition_scenes(c, train_frac, val_frac)

with open(os.path.join(dest_path, "train_scenes.txt"), "w") as f:
    for item in train_scenes:
        f.write("%s\n" % item)

with open(os.path.join(dest_path, "val_scenes.txt"), "w") as f:
    for item in val_scenes:
        f.write("%s\n" % item)

with open(os.path.join(dest_path, "test_scenes.txt"), "w") as f:
    for item in test_scenes:
        f.write("%s\n" % item)
