# this is best run on a 32 core machine to make the train dataset in under 30 minutes

# dataset with only vv imagery
# ceruleanml make-coco-dataset-no-context ../data/partitions/train/ ../train-no-context/ 1024
# gsutil -m rsync -r ../train-no-context-1024/ gs://ceruleanml/partitions/train-no-context-1024/
# rm -rf ../train-no-context-1024/

# ceruleanml make-coco-dataset-no-context ../data/partitions/val/ ../val-no-context/ 1024
# gsutil -m rsync -r ../val-no-context-1024/ gs://ceruleanml/partitions/val-no-context-1024/
# rm -rf ../val-no-context-1024/

# ceruleanml make-coco-dataset-no-context ../data/partitions/test/ ../test-no-context/ 1024
# gsutil -m rsync -r ../test-no-context-1024/ gs://ceruleanml/partitions/test-no-context-1024/
# rm -rf ../test-no-context-1024/

# # dataset with vv imagery + aux
ceruleanml make-coco-dataset-with-tiles ../data/partitions/train/ ../data/aux_datasets ../train-with-context-1024/ 1024
gsutil -m rsync -r ../train-with-context-1024/ gs://ceruleanml/partitions/train-with-context-1024/
rm -rf ../train-with-context-1024/

ceruleanml make-coco-dataset-with-tiles ../data/partitions/val/ ../data/aux_datasets ../val-with-context-1024/ 1024
gsutil -m rsync -r ../val-with-context-1024/ gs://ceruleanml/partitions/val-with-context-1024/
rm -rf ../val-with-context-1024/

ceruleanml make-coco-dataset-with-tiles ../data/partitions/test/ ../data/aux_datasets ../test-with-context-1024/ 1024
gsutil -m rsync -r ../test-with-context-1024/ gs://ceruleanml/partitions/test-with-context-1024/
rm -rf ../test-with-context-1024/

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

# # dataset with vv imagery + aux
ceruleanml make-coco-dataset-with-tiles ../data/partitions/train/ ../data/aux_datasets ../train-with-context-512/ 512
gsutil -m rsync -r ../train-with-context-512/ gs://ceruleanml/partitions/train-with-context-512/
rm -rf ../train-with-context-512/

ceruleanml make-coco-dataset-with-tiles ../data/partitions/val/ ../data/aux_datasets ../val-with-context-512/ 512
gsutil -m rsync -r ../val-with-context-512/ gs://ceruleanml/partitions/val-with-context-512/
rm -rf ../val-with-context-512/

ceruleanml make-coco-dataset-with-tiles ../data/partitions/test/ ../data/aux_datasets ../test-with-context-512/ 512
gsutil -m rsync -r ../test-with-context-512/ gs://ceruleanml/partitions/test-with-context-512/
rm -rf ../test-with-context-512/
