# this is best run on a 32 core machine to make the train dataset in under 30 minutes
python3.9 /root/work/ceruleanml/random_scene_select.py
gsutil cp /root/data/partitions/test_scenes.txt gs://ceruleanml/partitions/
gsutil cp /root/data/partitions/val_scenes.txt gs://ceruleanml/partitions/
gsutil cp /root/data/partitions/train_scenes.txt gs://ceruleanml/partitions/

# # dataset with vv imagery + aux
ceruleanml make-coco-dataset-with-tiles /root/data/partitions/test_scenes.txt /root/data/aux_datasets /root/data/partitions/test_tiles_context_1024/ 1024
gsutil -m rsync -r /root/data/partitions/test_tiles_context_1024/ gs://ceruleanml/partitions/test_tiles_context_1024/

ceruleanml make-coco-dataset-with-tiles /root/data/partitions/val_scenes.txt /root/data/aux_datasets /root/data/partitions/val_tiles_context_1024/ 1024
gsutil -m rsync -r /root/data/partitions/val_tiles_context_1024/ gs://ceruleanml/partitions/val_tiles_context_1024/

ceruleanml make-coco-dataset-with-tiles /root/data/partitions/train_scenes.txt /root/data/aux_datasets /root/data/partitions/train_tiles_context_1024/ 1024
gsutil -m rsync -r ../train_tiles_context_1024/ gs://ceruleanml/partitions/train_tiles_context_1024/

# dataset with only vv imagery
# ceruleanml make-coco-dataset-no-context ../data/partitions/train/ ../train-no-context-512/ 512
# gsutil -m rsync -r ../train-no-context-512/ gs://ceruleanml/partitions/train-no-context-512/
# rm -rf ../train-no-context-512/

# ceruleanml make-coco-dataset-no-context ../data/partitions/val/ ../val-no-context-512/ 512
# gsutil -m rsync -r ../val-no-context-512/ gs://ceruleanml/partitions/val-no-context-512/
# rm -rf ../val-no-context-512/

# ceruleanml make-coco-dataset-no-context ../data/partitions/test/ ../test-no-context-512/ 512
# gsutil -m rsync -r ../test-no-context-512/ gs://ceruleanml/partitions/test-no-context-512/
# rm -rf ../test-no-context-512/