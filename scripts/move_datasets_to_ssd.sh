mkdir -p /root/experiments/cv2

MEMTILE_SIZE=1024 # Setting MEMTILE_SIZE=0 generates a coco dataset with the full scenes, instead of tiling them first

mkdir -p /root/partitions/test_tiles_context_$MEMTILE_SIZE/
gsutil -m rsync -r gs://ceruleanml/partitions/test_tiles_context_$MEMTILE_SIZE/ /root/partitions/test_tiles_context_$MEMTILE_SIZE/

mkdir -p /root/partitions/val_tiles_context_$MEMTILE_SIZE/
gsutil -m rsync -r gs://ceruleanml/partitions/val_tiles_context_$MEMTILE_SIZE/ /root/partitions/val_tiles_context_$MEMTILE_SIZE/

mkdir -p /root/partitions/train_tiles_context_$MEMTILE_SIZE/
gsutil -m rsync -r gs://ceruleanml/partitions/train_tiles_context_$MEMTILE_SIZE/ /root/partitions/train_tiles_context_$MEMTILE_SIZE/