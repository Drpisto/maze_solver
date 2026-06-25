from collections import deque
import numpy as np


def find_pos(maze, value):
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == value:
                return i, j
    return None


def bfs(maze):
    start = find_pos(maze, 3)
    goal = find_pos(maze, 4)
    H, W = maze.shape
    queue = deque()
    queue.append((start[0], start[1], []))
    visited = {start}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c, path = queue.popleft()
        if (r, c) == goal:
            return path
        for action, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and maze[nr, nc] != 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, path + [action]))
    return []

# Cell types: 0=empty, 1=wall, 2=coin, 3=player, 4=end

# 5x5 Maze
maze_5x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 3, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 4, 1, 1]
])

# 7x7 Maze
maze_7x7 = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 3, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 4, 1],
    [1, 1, 1, 1, 1, 1, 1]
])

# 9x9 Maze
maze_9x9 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 3, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 4, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# 5x5 Maze - Variant 2
maze_5x5_v2 = np.array([
    [1, 1, 1, 1, 1],
    [1, 3, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 2, 1, 0, 1],
    [1, 1, 4, 2, 1]
])

# 5x5 Maze - Variant 3
maze_5x5_v3 = np.array([
    [1, 1, 1, 1, 1],
    [1, 3, 2, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 0, 2, 1],
    [1, 1, 4, 1, 1]
])

maze_5x5_v4 = np.array([
    [1, 2, 0, 1, 3],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 4],
])

# 7x7 Maze - Variant 2
maze_7x7_v2 = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 3, 0, 0, 1, 2, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 2, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 2, 1, 1, 1, 4, 1]
])

# 7x7 Maze - Variant 3
maze_7x7_v3 = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 3, 0, 1, 0, 2, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 2, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 2, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 4, 1]
])

# 9x9 Maze - Variant 2
maze_9x9_v2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 3, 0, 0, 1, 0, 2, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 2, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 2, 0, 1],
    [1, 2, 0, 0, 0, 0, 0, 4, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# 9x9 Maze - Variant 3
maze_9x9_v3 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 3, 0, 1, 0, 0, 0, 2, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 2, 0, 0, 1, 1, 1, 2, 1],
    [1, 0, 1, 0, 0, 0, 0, 4, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
])

mazes = [[maze_5x5, maze_5x5_v2, maze_5x5_v3,maze_5x5_v4], 
         [maze_7x7, maze_7x7_v2, maze_7x7_v3], 
         [maze_9x9, maze_9x9_v2, maze_9x9_v3]]