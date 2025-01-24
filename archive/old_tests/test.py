import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv

csv_file = "arcs_51_51_fixed.csv"

coordinates = []
energy = []

arc_dic = {}
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    # Skip header if present
    next(csv_reader, None)
    for row in csv_reader:
        choord, en = (row[1], row[2]), row[5]
        arc_dic[choord] = en

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = "arcs_51_51_fixed.csv"

# Step 1: Initialize the grid size and create an array for edge costs
grid_size = (51, 51)  # 51x51 grid
image_matrix = np.zeros((grid_size[0], grid_size[1]))

# Step 2: Parse CSV to fill the image matrix with energy costs
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    # Skip header if present
    next(csv_reader, None)
    for row in csv_reader:
        # Extract coordinates and energy cost
        x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
        energy_cost = float(row[5])

        # Set the corresponding pixel in the image matrix
        if (0 <= x1 < grid_size[0] and 0 <= y1 < grid_size[1] and
            0 <= x2 < grid_size[0] and 0 <= y2 < grid_size[1]):
            image_matrix[(x1 + x2) // 2, (y1 + y2) // 2] = energy_cost

# Step 3: Plot the image matrix as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(image_matrix, cmap="YlOrRd", square=True, cbar_kws={'label': 'Energy Cost'})
plt.title("Energy Cost Heatmap Between Nodes in Grid Network")
plt.xlabel("Node X Position")
plt.ylabel("Node Y Position")
plt.show()
