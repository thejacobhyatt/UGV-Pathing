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
spacing = 30
rows = l // spacing
cols = w // spacing
buffer = spacing // 2

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
    """creates 3D grid of nodes with all attributes neccesary for cost functions

    Args:
        rows (_type_): _description_
        cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    grid = [[[None for _ in range(2)] for _ in range(cols)] for _ in range(rows)]
    for j in range(rows):
        for i in range(cols):
            for z in range(2):
                grid[j][i][z] = Node(buffer + i * spacing,buffer + j * spacing)
    return grid

# elevation= (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation

def display_grid(super_grid, img=None):
    """Displays the grid over the image."""
    if img is not None:
        plt.imshow(img)
    
    for grid in super_grid:
        for row in grid:
            for node in row:
                plt.scatter(node.x, node.y, color="black", s=10)
    plt.show()


# Setup grid and neighbors
super_grid = setup(rows, cols)
print(super_grid)
# Display grid
display_grid(super_grid, img)
