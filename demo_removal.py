import argparse
from ludvig_uplift import LUDVIGUplift
from utils.config import Config

parser = argparse.ArgumentParser(description="Run the LUDVIGUplift model with configurable settings.")
args = parser.parse_args()

tag ="bonsai"
height = 1066
width = 1600

cfg = dict(
    colmap_dir=f"./dataset/{tag}/",
    gs_source=f"./dataset/{tag}/gs/point_cloud/iteration_30000/point_cloud.ply",
    height=height,
    width=width,
    save_visualizations=True
)
for model in ["dinov2", "clip", "eval"]:
    cfg["config"] = f"configs/demo_removal_{model}.yaml"
    cfg["tag"] = tag
    if model in ["clip", "dinov2"]:
        cfg["tag"] = tag + "/" + model
    model = LUDVIGUplift(Config(cfg))
    model.uplift()
    model.save()
