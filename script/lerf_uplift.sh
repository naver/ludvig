#!/bin/bash
scene=$1
it="30000"
src_path="lerf_ovs"
echo $scene
height=728
width=986
if [ "$scene" = "teatime" ]; then
    width=988; height=730
elif [ "$scene" = "ramen" ]; then
    width=988; height=731
elif [ "$scene" = "waldo_kitchen" ]; then
    width=985; height=725
fi

model=dinov2
python ludvig_uplift.py \
    --colmap_dir ./dataset/${src_path}/$scene \
    --gs_source ./dataset/${src_path}/$scene/gs/point_cloud/iteration_$it/point_cloud.ply \
    --config configs/lerf_${model}.yaml \
    --height ${height} \
    --width ${width} \
    --tag ${scene}/${model} \

model=clip
python ludvig_uplift.py \
    --colmap_dir ./dataset/${src_path}/$scene \
    --gs_source ./dataset/${src_path}/$scene/gs/point_cloud/iteration_$it/point_cloud.ply \
    --load_ply dinov2/gaussians.ply \
    --config configs/lerf_${model}.yaml \
    --height ${height} \
    --width ${width} \
    --tag ${scene}/${model} \
