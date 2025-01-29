import os

class EvaluationBase:
    def __init__(
        self,
        gaussian,
        features,
        render_fn,
        render_rgb,
        logdir,
        image_dir,
        colmap_cameras,
        scene,
        height,
        width,
    ):
        self.gaussian = gaussian
        self.features = features
        self.render_fn = render_fn
        self.render_rgb = render_rgb
        self.logdir = logdir
        self.image_dir = os.path.join(image_dir, "images")
        self.colmap_cameras = colmap_cameras
        self.scene = scene
        self.height = height
        self.width = width

    def __call__(self):
        return NotImplementedError("Base class.")
