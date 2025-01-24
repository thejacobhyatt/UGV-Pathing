import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 


class Node(): 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = []


# read in data from the images 
cwd = os.getcwd()
img = np.asarray(Image.open(cwd+'/imagery/Buckner_DEM.png'))
plt.imshow(img)
plt.show()
# initialize Nodes using that data 

# build the nnetwork of arcs properly 

# detection and energy functions across arcs

# write all data to CSV

# display nodes

