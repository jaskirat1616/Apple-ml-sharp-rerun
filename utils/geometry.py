"""
Geometry and transformation utilities.
"""

import numpy as np
from typing import Tuple, Optional


def rotation_matrix_y(degrees: float) -> np.ndarray:
    """
    Create rotation matrix around Y-axis.
    
    Args:
        degrees: Rotation angle in degrees
    
    Returns:
        3x3 rotation matrix
    """
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotate_points(points: np.ndarray, degrees: float, center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rotate 3D points around Y-axis.
    
    Args:
        points: Array of 3D points (N, 3)
        degrees: Rotation angle in degrees
        center: Rotation center (3,), or None to use origin
    
    Returns:
        Rotated points (N, 3)
    """
    R = rotation_matrix_y(degrees)
    
    if center is not None:
        # Translate to origin, rotate, translate back
        points_centered = points - center
        points_rotated = (R @ points_centered.T).T
        return points_rotated + center
    else:
        return (R @ points.T).T


def scale_points(points: np.ndarray, scale: float, center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Scale 3D points.
    
    Args:
        points: Array of 3D points (N, 3)
        scale: Scale factor
        center: Scale center (3,), or None to use origin
    
    Returns:
        Scaled points (N, 3)
    """
    if center is not None:
        # Translate to origin, scale, translate back
        points_centered = points - center
        points_scaled = points_centered * scale
        return points_scaled + center
    else:
        return points * scale


def translate_points(points: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Translate 3D points.
    
    Args:
        points: Array of 3D points (N, 3)
        translation: Translation vector (3,)
    
    Returns:
        Translated points (N, 3)
    """
    return points + translation


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box for points.
    
    Args:
        points: Array of 3D points (N, 3)
    
    Returns:
        Tuple of (min_corner, max_corner) where each is (3,)
    """
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    return min_corner, max_corner


def compute_center(points: np.ndarray) -> np.ndarray:
    """
    Compute center of points.
    
    Args:
        points: Array of 3D points (N, 3)
    
    Returns:
        Center point (3,)
    """
    return points.mean(axis=0)


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize points to unit sphere centered at origin.
    
    Args:
        points: Array of 3D points (N, 3)
    
    Returns:
        Tuple of (normalized_points, original_center, scale_factor)
    """
    center = compute_center(points)
    points_centered = points - center
    
    # Compute scale to fit in unit sphere
    max_dist = np.linalg.norm(points_centered, axis=1).max()
    if max_dist > 0:
        scale = 1.0 / max_dist
        points_normalized = points_centered * scale
    else:
        scale = 1.0
        points_normalized = points_centered
    
    return points_normalized, center, scale


def compute_distances(points: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute distances from points to reference point.
    
    Args:
        points: Array of 3D points (N, 3)
        reference: Reference point (3,), or None to use origin
    
    Returns:
        Array of distances (N,)
    """
    if reference is None:
        reference = np.zeros(3)
    return np.linalg.norm(points - reference, axis=1)

