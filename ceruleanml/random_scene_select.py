import glob
import os
import random
import subprocess

source_path = "./data-cv2/"
dest_path = "./data/partitions/"

classes = ["Coincident", "Infrastructure", "Old", "Recent", "Natural_Seep", "Ambiguous"]


def get_scenes(path):
    """Takes a path where annotations are written to. Returns scene folders within that path.
    Args:
        path (str): the path where annotations are written to.
    Returns:
        scenes (list): list of annotated S1 scenes.
    """
    scenes = [(x) for x in list(os.scandir(path)) if x.is_dir()]
    scenes = [(x.name) for x in scenes]
    return scenes

def partition_scenes(c, train_frac, test_frac):
    """Applies random selection of scenes for mutually exclusive partitions (train, validation, test). 
    Partition fractions must range from 0.0 to 1.0.
    Args:
        c (str): a target class.
        train_frac (float): Percent of items to allocate to training.
        test_frac (float): Percent of items to allocate to test. 
    Returns:
        train_scenes (list): List of train items.
        val_scenes_select (list): List of validation items.
        test_scenes_select (list): List of test items.
    """
    print("Selecting from class: ", c)
    if os.path.isdir(f"{source_path}/{c}"):
        if (
            not os.path.exists(f"{dest_path}train/{c}")
            and os.path.exists(f"{dest_path}val/{c}")
            and os.path.exists(f"{dest_path}test/{c}/")
        ):
            os.makedirs(f"{dest_path}train/{c}")
            os.makedirs(f"{dest_path}val/{c}")
            os.makedirs(f"{dest_path}test/{c}")

        scenes = get_scenes(f"{source_path}/{c}/*")
        len_train = int(len(scenes)*train_frac)
        len_test = int(len(scenes)*(test_frac))

        train_scenes = random.sample(scenes, len_train)
        scenes = [item for item in scenes if item not in train_scenes]
        test_scenes_select = random.sample(scenes, len_test)
        val_scenes_select = [item for item in scenes if item not in test_scenes_select]

    return train_scenes, val_scenes_select, test_scenes_select



for c in classes:
    train_scenes, val_scenes_select, test_scenes_select = partition_scenes(c)

with open(os.path.join(dest_path, "train_scenes.txt"), "w") as f:
    for item in train_scenes:
        f.write("%s\n" % item)

with open(os.path.join(dest_path, "val_scenes.txt"), "w") as f:
    for item in val_scenes_select:
        f.write("%s\n" % item)

with open(os.path.join(dest_path, "test_scenes.txt"), "w") as f:
    for item in test_scenes_select:
        f.write("%s\n" % item)

subprocess.run(
    [
        "rsync",
        "-r",
        "--files-from=./data/partitions/val_scenes.txt",
        "data-cv2/",
        "data/partitions/val/",
    ]
)
subprocess.run(
    [
        "rsync",
        "-r",
        "--files-from=./data/partitions/test_scenes.txt",
        "data-cv2/",
        "data/partitions/test/",
    ]
)
subprocess.run(
    [
        "rsync",
        "-r",
        "--files-from=./data/partitions/train_scenes.txt",
        "data-cv2/",
        "data/partitions/train/",
    ]
)
