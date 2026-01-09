# Examples

This directory contains example scripts demonstrating how to use the utility modules.

## 📚 Available Examples

### `example_visualization.py`

Basic visualization example showing how to:
- Load a PLY file using `load_gaussian_data()`
- Setup a Rerun viewer
- Visualize a 3D point cloud

**Usage:**
```bash
python examples/example_visualization.py -i output_test/IMG_4707.ply
```

### `example_navigation.py`

Navigation map building example showing how to:
- Load PLY data
- Detect ground plane
- Detect obstacles
- Build occupancy grid
- Visualize navigation data

**Usage:**
```bash
python examples/example_navigation.py -i output_test/IMG_4707.ply --resolution 0.5
```

## 💡 Writing Your Own Examples

All examples follow this pattern:

1. Add project root to Python path
2. Import from `utils` module
3. Use utility functions
4. Visualize with Rerun

**Template:**
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import (
    load_gaussian_data,
    # ... other utilities
)
```

## 🔧 Common Patterns

### Loading Data
```python
from utils import load_gaussian_data

data = load_gaussian_data("scene.ply")
positions = data['positions']
colors = data['colors']
scales = data['scales']
```

### Depth Rendering
```python
from utils import render_depth_map

depth_map, depth_colored = render_depth_map(
    positions, colors,
    resolution=(1280, 720),
    focal_length=1000
)
```

### Navigation
```python
from utils import extract_ground_plane, detect_obstacles

ground_points, ground_mask, _, ground_info = extract_ground_plane(positions)
obstacle_points, clusters = detect_obstacles(positions, ground_mask)
```

### Visualization Setup
```python
from utils import setup_complete_viewer_blueprint
import rerun as rr

rr.init("My Viewer", spawn=True)
blueprint = setup_complete_viewer_blueprint()
rr.send_blueprint(blueprint)
```

