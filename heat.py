import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data (replace 'arcs.csv' with your actual file path)
df = pd.read_csv('Buckner_arcs_12_12.csv', header=None)

# Extract necessary columns
source_nodes = df[1]
end_nodes = df[2]
weights = df[3]

# Define the 12x12 grid
grid_size = 12
grid = np.array(range(1, grid_size**2 + 1)).reshape(grid_size, grid_size)

# Create a mapping from node to (row, col) in the grid
node_to_pos = {grid[r, c]: (r, c) for r in range(grid_size) for c in range(grid_size)}

# Initialize a 12x12 weight matrix
weight_matrix = np.zeros((grid_size, grid_size))

# Fill the weight matrix based on node positions
for src, end, weight in zip(source_nodes, end_nodes, weights):
    if src in node_to_pos and end in node_to_pos:  # Ensure nodes exist in the grid
        r, c = node_to_pos[end]  # Use the END node position to visualize weight
        weight_matrix[r, c] = weight

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(weight_matrix, cmap='viridis', interpolation='gaussian')
plt.colorbar(label="Weight")
plt.xticks(ticks=range(grid_size), labels=range(1, grid_size + 1))
plt.yticks(ticks=range(grid_size), labels=range(1, grid_size + 1))
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
plt.title("Heatmap of Arc Weights in 12x12 Grid")

# Show the plot
plt.show()
