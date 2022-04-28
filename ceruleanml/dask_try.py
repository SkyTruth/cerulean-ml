import time

import dask


def foo():
    time.sleep(2)
    return "out"


def bar(out):
    time.sleep(2)
    return out


def main():
    start = time.time()
    coco_outputs = []
    class_folders = [1, 2, 3, 4]
    for i in class_folders:
        out = dask.delayed(foo)()
        coco_output = dask.delayed(bar)(out)
        coco_outputs.append(coco_output)
    coco_outputs = dask.compute(*coco_outputs, scheduler="processes")
    num_images = len(class_folders)
    print(f"Number of seconds for {num_images} images: {time.time() - start}")


if __name__ == "__main__":
    main()
