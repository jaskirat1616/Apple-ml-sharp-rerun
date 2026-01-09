"""
Input/Output utilities for file operations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load JSON file safely.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary with JSON data, or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
        indent: JSON indentation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image file as numpy array.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Numpy array of image, or None if loading fails
    """
    try:
        return np.array(Image.open(image_path))
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None


def save_image(image: np.ndarray, image_path: Path) -> bool:
    """
    Save numpy array as image.
    
    Args:
        image: Numpy array of image (H, W) or (H, W, 3)
        image_path: Path to save image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        Image.fromarray(image).save(image_path)
        return True
    except Exception as e:
        print(f"Error saving image to {image_path}: {e}")
        return False


def find_ply_files(directory: Path, pattern: str = "*.ply") -> List[Path]:
    """
    Find all PLY files in directory, sorted.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for PLY files
    
    Returns:
        List of PLY file paths, sorted
    """
    if not directory.exists():
        return []
    return sorted(list(directory.glob(pattern)))


def find_frame_images(directory: Path, pattern: str = "*.png") -> List[Path]:
    """
    Find all frame images in directory, sorted.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for image files
    
    Returns:
        List of image file paths, sorted
    """
    if not directory.exists():
        return []
    return sorted(list(directory.glob(pattern)))


def get_frame_number(filename: str) -> Optional[int]:
    """
    Extract frame number from filename (e.g., "frame_000123.png" -> 123).
    
    Args:
        filename: Filename with frame number
    
    Returns:
        Frame number, or None if not found
    """
    try:
        # Try to find frame_XXXXX pattern
        parts = Path(filename).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return None
    except Exception:
        return None

