import os, random, shutil

source_path = './data-cv2/' #os.getcwd()
dest_path = './data/partitions/'

val_scenes = 10
test_scenes = 3

val_scenes_select = []
test_scenes_select = []

classes = ['Coincident', 'Infrastructure', 'Old', 'Recent', 'Natural_Seep', 'Ambiguous']

def getRandomDir(path):
    randomDir = random.choice([(x) for x in list(os.scandir(path)) if x.is_dir()]).name
    return randomDir

for c in classes:
    print("Selecting from class: ", c)
    if os.path.isdir(f'{source_path}/{c}'):
        if not os.path.exists(f'{source_path}/{c}/val/') and os.path.exists(f'{source_path}/{c}/test/'):
            os.makedirs(f'{dest_path}/val/{c}')
            os.makedirs(f'{dest_path}/test/{c}')
        for i in range(val_scenes):
            val_scene = getRandomDir(f'{source_path}/{c}')
            if val_scene not in val_scenes_select:
                val_scenes_select.append(val_scene)
            else:
                val_scene = getRandomDir(f'{source_path}/{c}')
                val_scenes_select.append(val_scene)
            print("Validation scene selected: ", val_scene)
            shutil.copytree(f'{source_path}/{c}/{val_scene}', f'{dest_path}/val/{c}/{val_scene}', dirs_exist_ok=True)
        for i in range(test_scenes):
            test_scene = getRandomDir(f'{source_path}/{c}')
            if test_scene not in val_scenes_select and test_scene not in test_scenes_select:
                test_scenes_select.append(test_scene)
            else:
                test_scene = getRandomDir(f'{source_path}/{c}')
                test_scenes_select.append(test_scene)
            print("Test scene selected: ", test_scene)
            shutil.copytree(f'{source_path}/{c}/{test_scene}', f'{dest_path}/test/{c}/{test_scene}', dirs_exist_ok=True)
