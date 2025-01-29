cd dataset/
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip "stump/*"
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
unzip models.zip "stump/*" -d "stump/" && mv stump/stump stump/gs
