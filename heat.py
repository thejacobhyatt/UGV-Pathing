import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data (replace 'arcs.csv' with your actual file path)
df = pd.read_csv('Buckner_arcs_60_60.csv', header=None)

# Extract necessary columns
source_nodes = df[1]
end_nodes = df[2]
weights = df[5]

# Define the 12x12 grid
grid_size = 60
grid = np.array(range(1, grid_size**2 + 1)).reshape(grid_size, grid_size)
node_to_pos = {grid[r, c]: (r, c) for r in range(grid_size) for c in range(grid_size)}

weight_matrix = np.zeros((grid_size, grid_size))

for src, end, weight in zip(source_nodes, end_nodes, weights):
    if src in node_to_pos and end in node_to_pos:  
        r, c = node_to_pos[end]
        weight_matrix[r, c] = weight

plt.figure(figsize=(8, 8))
plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Weight")
plt.xticks(ticks=range(grid_size), labels=range(1, grid_size + 1))
plt.yticks(ticks=range(grid_size), labels=range(1, grid_size + 1))
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")

# Show the plot
plt.show()
