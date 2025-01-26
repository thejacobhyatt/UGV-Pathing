import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Constants 
SITUATION = "Buckner"
SPACING = 35
MAX_ELEVATION = 603

# Load in Images
cwd = os.getcwd()
sat_path = os.path.join(cwd, 'imagery', SITUATION+'.png')
sat_map = np.asarray(Image.open(sat_path))
elevation_path = os.path.join(cwd, 'imagery', SITUATION+'_DEM.png')
elevation_map = np.asarray(Image.open(elevation_path))
vegetation_path = os.path.join(cwd, 'imagery', SITUATION+'_NDVI.png')
vegetation_map = np.asarray(Image.open(vegetation_path))

# Derived Constants
l, w, h = sat_map.shape
rows = l // SPACING
cols = w // SPACING
buffer = SPACING // 2

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
        self.neighbors = {}

        self.rows,self.cols = self.position_to_grid(self.x, self.y)

    def get_elevation(self):
        self.e = round((elevation_map[self.x,self.y][0]/255)*MAX_ELEVATION,1)

    def get_vegetation(self):
        self.v = round((3 - 3*(vegetation_map[self.x,self.y][0]/255)), 3)

    def find_neighbors(self, grid, rows, cols):
        """Find neighbors in all cardinal and diagonal directions and the corresponding node in the opposite grid."""
        neighbors = []
        directions = [
            (0, 1),   # Down
            (1, 0),   # Right
            (0, -1),  # Up
            (-1, 0),  # Left
            (1, 1),   # Down-right
            (-1, -1), # Up-left
            (1, -1),  # Down-left
            (-1, 1)   # Up-right
        ]

        # Same grid neighbors
        for dx, dy in directions:
            nx, ny = self.rows + dx, self.cols + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbors.append(grid[self.z][ny][nx])

        # Opposite grid neighbor
        if self.z == 1: 
            neighbors.append(grid[0][self.cols][self.rows])
        else: 
            neighbors.append(grid[1][self.cols][self.rows])

        return neighbors

    def __str__(self):
        return f"Node({self.x}, {self.y}, {self.z}, {self.e}, {self.v})"
    
    def __repr__(self):
        return str(self)
    
    @staticmethod
    def position_to_grid(x, y, buffer=buffer, spacing=SPACING):
        """Convert x, y coordinates to grid row and column indices."""
        col = int((x - buffer) / spacing)
        row = int((y - buffer) / spacing)
        return row, col

class Arc():
    def __init__(self):
        pass
    

def setup(rows, cols):
    """creates 3D grid of nodes with all attributes neccesary for cost functions

    Args:
        rows (_type_): _description_
        cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    top_grid = [[None for _ in range(1) for _ in range(cols)] for _ in range(rows)]
    bottom_grid = [[None for _ in range(1) for _ in range(cols)] for _ in range(rows)]
    grid = [top_grid, bottom_grid]

    for z in range(len(grid)):  # Iterate over the top and bottom grids
        for j in range(rows):
            for i in range(cols):
                node = Node(buffer + i * SPACING, buffer + j * SPACING, z)
                node.get_elevation()
                node.get_vegetation()
                grid[z][j][i] = node  # Assign the node to the appropriate grid and position
    return grid


def display_grid(super_grid, img=None):
    """_summary_

    Args:
        super_grid (numpy.array): _description_
        img (image name, optional): _description_. Defaults to None.
    """
    if img is not None:
        plt.imshow(img)
    
    for grid in super_grid:
        for row in grid:
            for node in row:
                plt.scatter(node.x, node.y, color="black", s=5)
    plt.show()


# Setup grid and neighbors
super_grid = setup(rows, cols)
print(super_grid)
example_node = super_grid[1][1][1]  # Top grid, first node
neighbors = example_node.find_neighbors(super_grid, rows, cols)
print(example_node, ':' ,neighbors, len(neighbors))
# Display grid
display_grid(super_grid, elevation_map)
