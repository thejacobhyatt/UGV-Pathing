import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Grid dimensions
cwd = os.getcwd()
image_path = os.path.join(cwd, 'imagery', 'Buckner.png')
img = np.asarray(Image.open(image_path))
l, w, h = img.shape
spacing = 10
rows = l // spacing
cols = w // spacing
buffer = 5

class Node():
    def __init__(self, x, y, z=0, e=0, v=0):
        """Node class for pathfinding algorithm

        Args:
            x (int): x coord
            y (int): y coord
            z (int): 0 for not charging, 1 for charging
            e (float): elevation at that node
            v (float): vegetation value at that node
        """
        self.x = x
        self.y = y
        self.z = z
        self.e = e
        self.v = v
        self.neighbors = []

def setup(rows, cols):
    """Initializes the grid with Node objects."""
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for j in range(rows):
        for i in range(cols):
            grid[j][i] = Node(buffer + i * spacing,buffer + j * spacing)
    return grid

def add_neighbors(grid):
    """Connects nodes to their neighbors."""
    rows = len(grid)
    cols = len(grid[0])
    for j in range(rows):
        for i in range(cols):
            node = grid[j][i]
            if i > 0:  # Left neighbor
                node.neighbors.append(grid[j][i-1])
            if i < cols-1:  # Right neighbor
                node.neighbors.append(grid[j][i+1])
            if j > 0:  # Top neighbor
                node.neighbors.append(grid[j-1][i])
            if j < rows-1:  # Bottom neighbor
                node.neighbors.append(grid[j+1][i])

def display_grid(grid, img=None):
    """Displays the grid over the image."""
    if img is not None:
        plt.imshow(img)
    for row in grid:
        for node in row:
            plt.scatter(node.x, node.y, color="black", s=10)
    plt.show()


# Setup grid and neighbors
grid = setup(rows, cols)
add_neighbors(grid)

# Display grid
display_grid(grid, img)
