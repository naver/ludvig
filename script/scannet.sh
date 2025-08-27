#!/bin/bash
scene=$1
src_path="scannet"
echo $scene
height=968
width=1296

python ludvig_uplift.py \
    --colmap_dir ./dataset/${src_path}/$scene \
    --gs_source ./dataset/${src_path}/$scene/gs/point_cloud/iteration_30000/point_cloud.ply \
    --config configs/opengaussian_scannet.yaml \
    --height ${height} \
    --width ${width} \
    --tag ${scene} \
    --save_visualizations \
