#!/bin/bash
scene=$1
it="30000"
src_path="lerf_ovs"
height=728
width=986
if [ "$scene" = "teatime" ]; then
    width=988; height=730
elif [ "$scene" = "ramen" ]; then
    width=988; height=731
elif [ "$scene" = "waldo_kitchen" ]; then
    width=985; height=725
fi
echo $scene

optional_arg=${3:-}
python ludvig_clip.py \
    --colmap_dir ./dataset/${src_path}/$scene \
    --gs_source ./dataset/${src_path}/$scene/gs/point_cloud/iteration_$it/point_cloud.ply \
    --config configs/$2.yaml \
    --dino_features logs/lerf/${scene}/dinov2/features.npy \
    --clip_features logs/lerf/${scene}/clip/features.npy \
    --load_ply dinov2/gaussians.ply \
    --height ${height} \
    --width ${width} \
    --tag ${scene} \
    --n_runs 5 \
    ${optional_arg}
