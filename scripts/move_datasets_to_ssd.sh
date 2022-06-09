cd ~/work
mkdir ../train-no-context-512/
gsutil -m rsync -r  gs://ceruleanml/partitions/train-no-context-512/ ../train-no-context-512/

mkdir ../train-with-context-512/
gsutil -m rsync -r  gs://ceruleanml/partitions/train-with-context-512/ ../train-with-context-512/

mkdir ../val-no-context-512/
gsutil -m rsync -r  gs://ceruleanml/partitions/val-no-context-512/ ../val-no-context-512/

mkdir ../val-with-context-512/
gsutil -m rsync -r  gs://ceruleanml/partitions/val-with-context-512/ ../val-with-context-512/

cd ~
mkdir partititions
mv train* partitions/
mv val* partitions/
