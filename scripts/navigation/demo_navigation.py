#!/usr/bin/env python3
"""
DEMO: Autonomous Navigation with Rerun Visualization

This demo shows all the navigation features in action:
1. Load 3D scene
2. Detect ground and obstacles
3. Build occupancy grid
4. Plan paths
5. Visualize everything in Rerun

Run: python demo_navigation.py
"""

import numpy as np
from pathlib import Path
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
import torch
from heapq import heappush, heappop


def main():
    # Configuration
    PLY_FILE = Path("output_test/IMG_4707.ply")
    GRID_RESOLUTION = 0.5  # meters
    OBSTACLE_HEIGHT = 0.3  # meters
    
    print("=" * 70)
    print("🤖 AUTONOMOUS NAVIGATION DEMO")
    print("=" * 70)
    
    # Check if file exists
    if not PLY_FILE.exists():
        print(f"\n⚠️  File not found: {PLY_FILE}")
        print("\nAvailable files:")
        for f in Path("output_test").glob("*.ply"):
            print(f"  - {f}")
        print("\nUsage: python demo_navigation.py")
        return 1
    
    # Step 1: Load 3D Scene
    print(f"\n📂 Step 1: Loading 3D scene from {PLY_FILE.name}...")
    gaussians, metadata = load_ply(PLY_FILE)
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
    
    # Filter by opacity
    mask = opacities > 0.1
    positions = positions[mask]
    colors = colors[mask]
    
    # Convert colors
    if metadata.color_space == "linearRGB":
        colors_torch = torch.from_numpy(colors).float()
        colors = cs_utils.linearRGB2sRGB(colors_torch).numpy()
    
    print(f"  ✓ Loaded {len(positions):,} points")
    print(f"  Scene bounds:")
    print(f"    X: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
    print(f"    Y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")
    print(f"    Z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")
    
    # Step 2: Detect Ground Plane
    print(f"\n🌍 Step 2: Detecting ground plane...")
    ground_level = np.percentile(positions[:, 1], 10)  # Bottom 10% is likely ground
    ground_mask = np.abs(positions[:, 1] - ground_level) < OBSTACLE_HEIGHT
    ground_points = positions[ground_mask]
    
    print(f"  ✓ Ground level: {ground_level:.2f} m")
    print(f"  ✓ Ground points: {len(ground_points):,}")
    
    # Step 3: Detect Obstacles
    print(f"\n🚧 Step 3: Detecting obstacles...")
    obstacle_mask = ~ground_mask & (positions[:, 1] > (ground_level + OBSTACLE_HEIGHT))
    obstacle_points = positions[obstacle_mask]
    
    print(f"  ✓ Obstacle points: {len(obstacle_points):,}")
    
    # Simple clustering
    if len(obstacle_points) > 0:
        from scipy.spatial import KDTree
        tree = KDTree(obstacle_points)
        clusters = []
        visited = np.zeros(len(obstacle_points), dtype=bool)
        
        for i in range(min(len(obstacle_points), 1000)):  # Check first 1000 points
            if visited[i]:
                continue
            indices = tree.query_ball_point(obstacle_points[i], 2.0)
            if len(indices) > 10:  # Significant cluster
                visited[indices] = True
                cluster_pts = obstacle_points[indices]
                clusters.append({
                    'center': cluster_pts.mean(axis=0),
                    'size': (cluster_pts.max(axis=0) - cluster_pts.min(axis=0)),
                    'points': len(indices)
                })
        
        print(f"  ✓ Found {len(clusters)} obstacle clusters")
    else:
        clusters = []
    
    # Step 4: Build Occupancy Grid
    print(f"\n📊 Step 4: Building occupancy grid (resolution: {GRID_RESOLUTION}m)...")
    
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_z, max_z = positions[:, 2].min(), positions[:, 2].max()
    
    nx = int((max_x - min_x) / GRID_RESOLUTION) + 1
    nz = int((max_z - min_z) / GRID_RESOLUTION) + 1
    
    occupancy_grid = np.zeros((nx, nz))
    
    # Mark occupied cells (anything above ground)
    for pos in positions:
        if pos[1] > (ground_level + OBSTACLE_HEIGHT):
            i = int((pos[0] - min_x) / GRID_RESOLUTION)
            j = int((pos[2] - min_z) / GRID_RESOLUTION)
            if 0 <= i < nx and 0 <= j < nz:
                occupancy_grid[i, j] = 1
    
    free_cells = (occupancy_grid == 0).sum()
    occupied_cells = (occupancy_grid == 1).sum()
    
    print(f"  ✓ Grid size: {nx} x {nz}")
    print(f"  ✓ Free cells: {free_cells:,} ({100*free_cells/occupancy_grid.size:.1f}%)")
    print(f"  ✓ Occupied cells: {occupied_cells:,} ({100*occupied_cells/occupancy_grid.size:.1f}%)")
    print(f"  ✓ Navigable area: {free_cells * GRID_RESOLUTION**2:.1f} m²")
    
    # Step 5: Plan Path (if enough free space)
    path_3d = None
    if free_cells > 100:
        print(f"\n🗺️  Step 5: Planning navigation path...")
        
        # Find free cells for start and goal
        free_indices = np.argwhere(occupancy_grid == 0)
        if len(free_indices) > 10:
            # Pick start near bottom-left, goal near top-right
            start_idx = free_indices[0]
            goal_idx = free_indices[-1]
            
            print(f"  Start grid cell: {start_idx}")
            print(f"  Goal grid cell: {goal_idx}")
            
            # Simple A* path finding
            def heuristic(a, b):
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
            def neighbors(cell):
                i, j = cell
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < nz and occupancy_grid[ni, nj] == 0:
                            yield (ni, nj)
            
            # A* search
            start = tuple(start_idx)
            goal = tuple(goal_idx)
            open_set = [(0, start)]
            came_from = {}
            g_score = {start: 0}
            
            path_found = False
            while open_set and len(came_from) < 10000:  # Limit search
                _, current = heappop(open_set)
                
                if current == goal:
                    # Reconstruct path
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path = path[::-1]
                    path_found = True
                    
                    # Convert to 3D coordinates
                    path_3d = []
                    for i, j in path:
                        x = min_x + i * GRID_RESOLUTION
                        z = min_z + j * GRID_RESOLUTION
                        y = ground_level + 0.5  # Slightly above ground
                        path_3d.append([x, y, z])
                    path_3d = np.array(path_3d)
                    
                    print(f"  ✓ Path found with {len(path)} waypoints")
                    print(f"  ✓ Path length: ~{len(path) * GRID_RESOLUTION:.1f} m")
                    break
                
                for neighbor in neighbors(current):
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heappush(open_set, (f_score, neighbor))
            
            if not path_found:
                print(f"  ⚠️  No path found (obstacles may be blocking)")
    else:
        print(f"\n⚠️  Step 5: Not enough free space for path planning")
    
    # Step 6: Visualize in Rerun
    print(f"\n🎨 Step 6: Launching Rerun visualization...")
    print("\nRerun will open with:")
    print("  • 3D View: Full scene with color-coded elements")
    print("  • 2D View: Top-down occupancy grid")
    print("  • Green: Ground/navigable space")
    print("  • Red: Obstacles")
    print("  • Cyan: Planned navigation path")
    print("  • Orange: Obstacle bounding boxes")
    
    # Initialize Rerun
    rr.init("🤖 Autonomous Navigation Demo", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Set up blueprint
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Vertical(
            rr.blueprint.Spatial3DView(name="🎯 3D Navigation Map", origin="world"),
            rr.blueprint.Spatial2DView(name="🗺️  Occupancy Grid (Top-Down)", origin="grid"),
            row_shares=[2, 1]
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)
    
    # Log full scene (gray)
    rr.log(
        "world/scene/full_cloud",
        rr.Points3D(
            positions=positions,
            colors=[[0.6, 0.6, 0.6]] * len(positions),
            radii=[0.1] * len(positions)
        )
    )
    
    # Log ground (green)
    if len(ground_points) > 0:
        rr.log(
            "world/navigation/ground",
            rr.Points3D(
                positions=ground_points,
                colors=[[0.0, 1.0, 0.0]] * len(ground_points),
                radii=[0.2] * len(ground_points)
            )
        )
    
    # Log obstacles (red)
    if len(obstacle_points) > 0:
        rr.log(
            "world/navigation/obstacles",
            rr.Points3D(
                positions=obstacle_points,
                colors=[[1.0, 0.0, 0.0]] * len(obstacle_points),
                radii=[0.15] * len(obstacle_points)
            )
        )
    
    # Log obstacle bounding boxes (orange)
    for i, cluster in enumerate(clusters):
        center = cluster['center']
        size = cluster['size']
        rr.log(
            f"world/obstacles/bbox_{i}",
            rr.Boxes3D(
                half_sizes=[size / 2],
                centers=[center],
                colors=[[255, 150, 0]]
            )
        )
    
    # Log occupancy grid
    grid_viz = np.zeros((nx, nz, 3), dtype=np.uint8)
    grid_viz[occupancy_grid == 0] = [0, 255, 0]  # Green = free
    grid_viz[occupancy_grid == 1] = [255, 0, 0]  # Red = occupied
    
    rr.log("grid/occupancy", rr.Image(grid_viz))
    
    # Log path if found
    if path_3d is not None:
        rr.log(
            "world/navigation/path",
            rr.LineStrips3D(
                [path_3d],
                colors=[[0, 255, 255]],
                radii=[0.3]
            )
        )
        
        # Mark start and goal
        rr.log(
            "world/navigation/start",
            rr.Points3D(
                positions=[path_3d[0]],
                colors=[[0, 255, 0]],
                radii=[1.0]
            )
        )
        rr.log(
            "world/navigation/goal",
            rr.Points3D(
                positions=[path_3d[-1]],
                colors=[[255, 0, 255]],
                radii=[1.0]
            )
        )
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"  • Scene: {len(positions):,} points")
    print(f"  • Ground: {len(ground_points):,} points")
    print(f"  • Obstacles: {len(obstacle_points):,} points ({len(clusters)} clusters)")
    print(f"  • Grid: {nx}x{nz} cells ({GRID_RESOLUTION}m resolution)")
    print(f"  • Navigable area: {free_cells * GRID_RESOLUTION**2:.1f} m²")
    if path_3d is not None:
        print(f"  • Path: {len(path_3d)} waypoints (~{len(path_3d) * GRID_RESOLUTION:.1f}m)")
    
    print("\n🎮 Rerun Controls:")
    print("  • Left click + drag: Rotate 3D view")
    print("  • Shift + Left click: Pan view")
    print("  • Mouse wheel: Zoom in/out")
    print("  • Double click: Reset camera")
    
    print("\n💡 Try These Next:")
    print("  1. python demo_navigation.py  # Run again with different file")
    print("  2. python build_navigation_map.py -i your_file.ply")
    print("  3. python create_camera_path.py -i your_file.ply --orbit")
    print("  4. python apply_depth_effects.py -i your_file.ply --fog")
    
    print("\nPress Ctrl+C to exit.")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

