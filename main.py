import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

rows = 10
cols = 10


class Node(): 
    def __init__(self, x, y, z, e, v):
        """Node class for pathfinding algorithm

        Args:
            x (int): x choord
            y (int): y choord
            z (int): 0 for notcharging, 1 for charging
            e (float): elevation at that node
            v (float): vegetation value at that node
        """
        self.x = x
        self.y = y
        self.z = z
        self.e = e
        self.v = v
        self.neighbors = []


# read in data from the images 
cwd = os.getcwd()
img = np.asarray(Image.open(cwd+'/imagery/Buckner_DEM.png'))
plt.imshow(img)
plt.show()


def setup(rows, cols):
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for j in range(rows):
        for i in range(cols):
            grid[j][i] = Cell(i, j)

    return grid
# initialize Nodes using that data 

# build the nnetwork of arcs properly 

# detection and energy functions across arcs

# write all data to CSV

# display nodes

