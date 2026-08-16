# coordinate_transformer.py:
# Converts banana positions & orientations from the Realsense optical frame to the Franka base frame
# Uses the hand-eye calibration stored in "franka camera tf.txt"

from __future__ import annotations

from typing import Iterable

import numpy as np


# Camera optical frame to Franka base frame transformation matrix
T_CAM_TO_ROBOT = np.array([
    [0.0124, -0.9864, -0.1640, 0.4471],
    [-0.9996, -0.0078, -0.0284, -0.4186],
    [0.0267, 0.1643, -0.9860, 1.2531],
    [0.0000, 0.0000, 0.0000, 1.0000],
], dtype=np.float64)

# The banana is on a platform 12 cm above the original working surface
# This is applied after the camera-to-Franka transformation
BANANA_PLATFORM_Z_OFFSET = 0.11


def _vector(values: Iterable[float], length: int, name: str) -> np.ndarray:
    """Convert an input to a finite one-dimensional vector."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def transform_point(coords: Iterable[float]) -> np.ndarray:
    """Transform one [x, y, z] point from camera frame to Franka base frame."""
    point_camera = _vector(coords, 3, "coords")
    point_homogeneous = np.append(point_camera, 1.0)
    point_robot = (T_CAM_TO_ROBOT @ point_homogeneous)[:3]

    # Add the platform height after completing the normal transformation
    point_robot[2] += BANANA_PLATFORM_Z_OFFSET
    return point_robot


def transform_points(coords: Iterable[Iterable[float]]) -> np.ndarray:
    """Transform an array of camera-frame points with shape (N, 3)."""
    points_camera = np.asarray(coords, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if not np.all(np.isfinite(points_camera)):
        raise ValueError("coords contains a non-finite value")

    homogeneous = np.column_stack((points_camera, np.ones(len(points_camera))))
    points_robot = (T_CAM_TO_ROBOT @ homogeneous.T).T[:, :3]

    # Apply the same platform offset to every transformed corner
    points_robot[:, 2] += BANANA_PLATFORM_Z_OFFSET
    return points_robot


def transform_direction(direction: Iterable[float]) -> np.ndarray:
    """Rotate a direction vector into the Franka frame without translating it."""
    direction_camera = _vector(direction, 3, "direction")
    direction_robot = T_CAM_TO_ROBOT[:3, :3] @ direction_camera
    magnitude = np.linalg.norm(direction_robot)
    if magnitude < 1e-9:
        raise ValueError("direction must not be a zero vector")
    return direction_robot / magnitude


def transform_object_angle(angle_camera: float) -> float:
    """Convert the banana's image-plane angle to a Franka-base XY heading."""
    # Image x points right & image y points down in the Realsense optical frame
    direction_camera = [np.cos(angle_camera), np.sin(angle_camera), 0.0]
    direction_robot = transform_direction(direction_camera)
    return float(np.arctan2(direction_robot[1], direction_robot[0]))


# Kept with the original function name so older SALAD scripts can still call it
def franka_coordTransform(coords: Iterable[float]) -> list[float]:
    return transform_point(coords).tolist()
