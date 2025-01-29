import numpy as np
from gaussiansplatting.scene.cameras import Simple_Camera
from .camera_multinerf import generate_interpolated_path


def interpolate_cameras(cameras, n_interp=50, smoothness=0.2, spline_degree=5):
    """
    Returns an array of cameras, interpolated with bsplines from the original cameras
    """
    # interpolate poses using B-Splines
    poses = np.ndarray((len(cameras), 3, 4))
    for i, camera in enumerate(cameras):
        poses[i, :3, :3] = camera.R
        poses[i, :3, 3] = camera.T
    interpolated_poses = generate_interpolated_path(
        poses, n_interp, smoothness=smoothness, spline_degree=spline_degree
    )

    # Create GS cameras from those poses
    interpolated_cameras = [
        Simple_Camera(
            colmap_id=-1,
            R=pose[:3, :3],
            T=pose[:3, 3],
            FoVx=cameras[0].FoVx,
            FoVy=cameras[0].FoVy,
            h=cameras[0].image_height,
            w=cameras[0].image_width,
            image_name="interpolated_image",
            uid="interpolated_camera",
            scale=1.0,
            data_device="cuda",
        )
        for pose in interpolated_poses
    ]
    return interpolated_cameras
