import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from exenf_alog import direction_of_travel, exenf_cost

# Constants 
SITUATION = "Buckner"
SPACING = 40
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
N = rows * cols * 2

class Node():
    def __init__(self, node_id, x, y, z=0, e=0, v=0):
        """Node class for pathfinding algorithm

        Args:
            x (int): x coord
            y (int): y coord
            z (int): 0 for not charging, 1 for charging
            e (float): elevation at that node
            v (float): vegetation value at that node
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.e = e
        self.v = v
        self.neighbors = {}

        self.row,self.col = self.position_to_grid(self.x, self.y)

    def get_elevation(self):
        self.e = round((elevation_map[self.x,self.y][0]/255)*MAX_ELEVATION,1)

    def get_vegetation(self):
        self.v = round((3 - 3*(vegetation_map[self.x,self.y][0]/255)), 3)

    def find_neighbors(self, grid, rows, cols):
        """Find neighbors in all cardinal and diagonal directions and the corresponding node in the opposite grid."""
        neighbors = []
        directions = [
            (1, 0),   # Right
            (1, 1),   # Down-right
            (0, 1),   # Down
            (-1, 1),   # Up-right
            (-1, 0),  # Left
            (-1, -1), # Up-left
            (0, -1),  # Up
            (1, -1)  # Down-left
        ]

        # Create Neighbor
        for dx, dy in directions:
            nx, ny = self.col + dx, self.row + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                # Avoid including the current node itself as a neighbor
                if nx != self.col or ny != self.row:  # Check if it's not the current node
                    neighbors.append(grid[self.z][ny][nx])

        # Add Vertical Neighbor
        if self.z == 1: 
            neighbors.append(grid[0][self.row][self.col])
        else:
            neighbors.append(grid[1][self.row][self.col])

        return neighbors

    
    def calculate_arc(self, neighbor):
        """Calculate arc properties between this node and a neighbor."""
        position_self = np.array([self.x, self.y, self.e])
        position_neighbor = np.array([neighbor.x, neighbor.y, neighbor.e])

        travel_time = 60 #TODO: Should be a function of distance

        if self.z == 1 and neighbor.z == 1:
            mode_of_travel='charging'
            movement_code=1
        elif self.z == 0 and neighbor.z == 0:
            mode_of_travel='charged'
            movement_code=0
        elif self.z == 1 and neighbor.z == 0:
            mode_of_travel='charged'
            movement_code=0
        elif self.z == 0 and neighbor.z == 1:
            mode_of_travel='charging'
            movement_code=1

        # Risk level (placeholder logic for visual/audio detection)
        visual_detection = 0.5  # Replace with `get_visual_detection`
        audio_detection = 0.3  # Replace with `get_audio_detection`
        risk_level = max(visual_detection, audio_detection)

        # Energy cost
        if np.array_equal(position_self, position_neighbor):
            energy_cost = 0
            travel_time = 0 #TODO: Should be a function of distance
        elif mode_of_travel == 'charging':
            energy_cost = 100
        elif mode_of_travel == 'charged':
            energy_cost = 150

        return [risk_level, travel_time, energy_cost, movement_code]

    def direction_of_travel(initial_point, final_point):
        x1, y1, _ = initial_point
        x2, y2, _ = final_point
        
        dx = x2 - x1
        dy = y2 - y1

        heading = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        return heading


    def exenf_cost(params, fcns):
        """Simulated exergy/energy cost calculation."""
        return 10, 5, "success"  # Replace with actual logic

    def __str__(self):
        return f"{self.id}"
    
    def __repr__(self):
        return str(self)

    @staticmethod
    def position_to_grid(x, y, buffer=buffer, spacing=SPACING):
        """Convert x, y coordinates to grid row and column indices."""
        col = int((x - buffer) / spacing)
        row = int((y - buffer) / spacing)
        return row, col

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
    node_id = 1

    # Iterate over the top and bottom grids
    for z in range(len(grid)):  # Iterate over the two grids (top and bottom)
        for j in range(rows):  # Iterate over rows
            for i in range(cols):  # Iterate over columns
                # Create a node with a unique ID and appropriate position
                node = Node(node_id, buffer + i * SPACING, buffer + j * SPACING, z)
                node.get_elevation()  # Assuming this method gets elevation data
                node.get_vegetation()  # Assuming this method gets vegetation data
                grid[z][j][i] = node  # Assign the node to the appropriate grid and position
                node_id += 1  # Increment the node ID for the next node
    return grid

def get_arcs(grid):
    """Calculate arcs between nodes and their neighbors."""
    arc_dict = {}
    arc_id = 1

    for z in range(len(grid)):  # Iterate through top and bottom grids
        for row in grid[z]:
            for node in row:
                neighbors = node.find_neighbors(grid, rows, cols)
                for neighbor in neighbors:
                    properties = node.calculate_arc(neighbor)
                    arc_dict[arc_id] = [node.id, neighbor.id] + properties
                    arc_id += 1

    return arc_dict

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
arc_dictionary = get_arcs(super_grid)
for key, value in arc_dictionary.items():
    print(f"{key}, {value}")

# print(super_grid[1][2][2])
# display_grid(super_grid, sat_map)