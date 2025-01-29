import argparse
from ludvig_uplift import LUDVIGUplift
from utils.config import Config

parser = argparse.ArgumentParser(description="Run the LUDVIGUplift model with configurable settings.")
parser.add_argument("--rgb", action="store_true", help="Replace 'demo_dino.yaml' with 'demo_rgb.yaml'")
args = parser.parse_args()

ext = "_rgb" * args.rgb
cfg = dict(
    colmap_dir="./dataset/stump/",
    gs_source="./dataset/stump/gs/point_cloud/iteration_30000/point_cloud.ply",
    config=f"configs/demo{ext}.yaml",
    height=1060,
    width=1600,
    save_visualizations=True
)

model = LUDVIGUplift(Config(cfg))
model.uplift()
model.save()
