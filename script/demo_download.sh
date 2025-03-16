cd dataset/
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
unzip 360_v2.zip "stump/*"
unzip models.zip "stump/*" -d "stump/" && mv stump/stump stump/gs
unzip 360_v2.zip "bonsai/*"
unzip models.zip "bonsai/*" -d "bonsai/" && mv bonsai/bonsai bonsai/gs
