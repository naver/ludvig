# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera pose and ray generation utility functions."""

import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union

# from internal import configs
# from internal import math
# from internal import stepfun
# from internal import utils
import numpy as np
import scipy

_Array = np.ndarray


def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


# Constants for generate_spiral_path():
NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.


def generate_spiral_path(poses: np.ndarray,
                         bounds: np.ndarray,
                         n_frames: int = 120,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of conservative near and far bounds in disparity space.
  near_bound = bounds.min() * NEAR_STRETCH
  far_bound = bounds.max() * FAR_STRETCH
  # All cameras will point towards the world space point (0, 0, -focal).
  focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = average_pose(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  # Just make sure it's it in the [-1, 1]^3 cube
  scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  poses_recentered[:, :3, 3] *= scale_factor
  transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  return poses_recentered, transform


# def generate_ellipse_path(poses: np.ndarray,
#                           n_frames: int = 120,
#                           const_speed: bool = True,
#                           z_variation: float = 0.,
#                           z_phase: float = 0.) -> np.ndarray:
#   """Generate an elliptical render path based on the given poses."""
#   # Calculate the focal point for the path (cameras point toward this).
#   center = focus_point_fn(poses)
#   # Path height sits at z=0 (in middle of zero-mean capture pattern).
#   offset = np.array([center[0], center[1], 0])

#   # Calculate scaling for ellipse axes based on input camera positions.
#   sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
#   # Use ellipse that is symmetric about the focal point in xy.
#   low = -sc + offset
#   high = sc + offset
#   # Optional height variation need not be symmetric
#   z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
#   z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

#   def get_positions(theta):
#     # Interpolate between bounds with trig functions to get ellipse in x-y.
#     # Optionally also interpolate in z to change camera height along path.
#     return np.stack([
#         low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
#         low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
#         z_variation * (z_low[2] + (z_high - z_low)[2] *
#                        (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
#     ], -1)

#   theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
#   positions = get_positions(theta)

#   if const_speed:
#     # Resample theta angles so that the velocity is closer to constant.
#     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
#     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
#     positions = get_positions(theta)

#   # Throw away duplicated last position.
#   positions = positions[:-1]

#   # Set path's up vector to axis closest to average of input pose up vectors.
#   avg_up = poses[:, :3, 1].mean(0)
#   avg_up = avg_up / np.linalg.norm(avg_up)
#   ind_up = np.argmax(np.abs(avg_up))
#   up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

#   return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1):
  """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

  def poses_to_points(poses, dist):
    """Converts from pose matrices to (position, lookat, up) format."""
    pos = poses[:, :3, -1]
    lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
    up = poses[:, :3, -1] + dist * poses[:, :3, 1]
    return np.stack([pos, lookat, up], 1)

  def points_to_poses(points):
    """Converts from (position, lookat, up) format to pose matrices."""
    return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

  def interp(points, n, k, s):
    """Runs multidimensional B-spline interpolation on the input points."""
    sh = points.shape
    pts = np.reshape(points, (sh[0], -1))
    k = min(k, sh[0] - 1)
    tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
    u = np.linspace(0, 1, n, endpoint=False)
    new_points = np.array(scipy.interpolate.splev(u, tck))
    new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
    return new_points

  points = poses_to_points(poses, dist=rot_weight)
  new_points = interp(points,
                      n_interp * (points.shape[0] - 1),
                      k=spline_degree,
                      s=smoothness)
  return points_to_poses(new_points)


def interpolate_1d(x: np.ndarray,
                   n_interp: int,
                   spline_degree: int,
                   smoothness: float) -> np.ndarray:
  """Interpolate 1d signal x (by a factor of n_interp times)."""
  t = np.linspace(0, 1, len(x), endpoint=True)
  tck = scipy.interpolate.splrep(t, x, s=smoothness, k=spline_degree)
  n = n_interp * (len(x) - 1)
  u = np.linspace(0, 1, n, endpoint=False)
  return scipy.interpolate.splev(u, tck)


def create_render_spline_path(
    config,
    image_names,
    poses,
    exposures,
):
  """Creates spline interpolation render path from subset of dataset poses.

  Args:
    config: configs.Config object.
    image_names: either a directory of images or a text file of image names.
    poses: [N, 3, 4] array of extrinsic camera pose matrices.
    exposures: optional list of floating point exposure values.

  Returns:
    spline_indices: list of indices used to select spline keyframe poses.
    render_poses: array of interpolated extrinsic camera poses for the path.
    render_exposures: optional list of interpolated exposures for the path.
  """
  if utils.isdir(config.render_spline_keyframes):
    # If directory, use image filenames.
    keyframe_names = sorted(utils.listdir(config.render_spline_keyframes))
  else:
    # If text file, treat each line as an image filename.
    with utils.open_file(config.render_spline_keyframes, 'r') as fp:
      # Decode bytes into string and split into lines.
      keyframe_names = fp.read().decode('utf-8').splitlines()
  # Grab poses corresponding to the image filenames.
  spline_indices = np.array(
      [i for i, n in enumerate(image_names) if n in keyframe_names])
  keyframes = poses[spline_indices]
  render_poses = generate_interpolated_path(
      keyframes,
      n_interp=config.render_spline_n_interp,
      spline_degree=config.render_spline_degree,
      smoothness=config.render_spline_smoothness,
      rot_weight=.1)
  if config.render_spline_interpolate_exposure:
    if exposures is None:
      raise ValueError('config.render_spline_interpolate_exposure is True but '
                       'create_render_spline_path() was passed exposures=None.')
    # Interpolate per-frame exposure value.
    log_exposure = np.log(exposures[spline_indices])
    # Use aggressive smoothing for exposure interpolation to avoid flickering.
    log_exposure_interp = interpolate_1d(
        log_exposure,
        config.render_spline_n_interp,
        spline_degree=5,
        smoothness=20)
    render_exposures = np.exp(log_exposure_interp)
  else:
    render_exposures = None
  return spline_indices, render_poses, render_exposures
