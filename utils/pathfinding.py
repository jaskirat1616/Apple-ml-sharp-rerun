"""
Pathfinding algorithms for navigation in occupancy grids.
"""

from heapq import heappush, heappop


def find_free_paths(occupancy_grid, start_cell, goal_cell):
    """
    Simple A* path finding on occupancy grid.
    
    Args:
        occupancy_grid: 2D array where 0=free, 1=occupied
        start_cell: Tuple of (i, j) grid coordinates for start
        goal_cell: Tuple of (i, j) grid coordinates for goal
    
    Returns:
        List of grid coordinates (i, j) representing the path, or None if no path found
    """
    nx, nz = occupancy_grid.shape
    
    def heuristic(a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def neighbors(cell):
        """Get valid neighboring cells (8-connected grid)."""
        i, j = cell
        # 8-connected grid
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < nz and occupancy_grid[ni, nj] == 0:
                    yield (ni, nj)
    
    # A* search
    open_set = []
    heappush(open_set, (0, start_cell))
    came_from = {}
    g_score = {start_cell: 0}
    f_score = {start_cell: heuristic(start_cell, goal_cell)}
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal_cell:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for neighbor in neighbors(current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_cell)
                heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

