import glob
import os
import random
import subprocess

source_path = "./data-cv2/"
dest_path = "./data/partitions/"

val_scenes = 10
test_scenes = 3

train_scenes = []
val_scenes_select = []
test_scenes_select = []

classes = ["Coincident", "Infrastructure", "Old", "Recent", "Natural_Seep", "Ambiguous"]


def getRandomDir(path):
    """Takes a path where annotations are written to. Randomly selects scene folders within that path.
    Args:
        path (str): the path where annotations are written to.
    Returns:
        randomly selected scene (str): an annotated S1 scene.
    """
    randomDir = random.choice([(x) for x in list(os.scandir(path)) if x.is_dir()]).name
    return randomDir


def partition_scenes(c):
    """Applies random selection of scenes for mutually exclusive partitions.
    Args:
        c (str): a target class.
    Returns:
        n/a.
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
        for i in range(val_scenes):
            val_scene = getRandomDir(f"{source_path}/{c}")
            val_scene = f"{c}/{val_scene}"
            if val_scene not in val_scenes_select:
                val_scenes_select.append(val_scene)
            else:
                val_scene = getRandomDir(f"{source_path}/{c}")
                val_scene = f"{c}/{val_scene}"
                val_scenes_select.append(val_scene)
            print("Validation scene selected: ", val_scene)
        for i in range(test_scenes):
            test_scene = getRandomDir(f"{source_path}/{c}")
            test_scene = f"{c}/{test_scene}"
            if (
                test_scene not in val_scenes_select
                and test_scene not in test_scenes_select
            ):
                test_scenes_select.append(test_scene)
            else:
                test_scene = getRandomDir(f"{source_path}/{c}")
                test_scene = f"{c}/{test_scene}"
                test_scenes_select.append(test_scene)
            print("Test scene selected: ", test_scene)
        scenes = glob.glob(f"{source_path}/{c}/*")
        for train_scene in scenes:
            train_scene = train_scene.split("/")[
                -1
            ]  # os.path.splitext(train_scene)[-1]
            if (
                train_scene not in val_scenes_select
                and train_scene not in test_scenes_select
            ):
                train_scene = f"{c}/{train_scene}"
                train_scenes.append(train_scene)
                print("Train scene: ", train_scene)
    return


for c in classes:
    partition_scenes(c)

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
