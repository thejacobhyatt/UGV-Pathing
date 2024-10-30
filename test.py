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

grid_size = (51, 51)  # 51x51 grid
num_nodes = grid_size[0] * grid_size[1]

# Step 3: Create the graph and add nodes
G = nx.grid_2d_graph(grid_size[0], grid_size[1])

# Step 4: Assign actual energy costs to each edge
for (u, v) in G.edges():
    if (u, v) in arc_dic:
        G.edges[u, v]['weight'] = arc_dic[(u, v)]
    else:
        G.edges[u, v]['weight'] = 0  # If no energy cost specified, set it to 0 or some default

# Step 5: Prepare position for visualization
pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Arrange nodes in grid layout

# Step 6: Draw the network graph
plt.figure(figsize=(12, 12))
# Draw nodes
nx.draw(G, pos, node_size=300, node_color="lightblue", with_labels=True)
# Draw edge labels with weights (costs)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
# Draw edges with color intensities based on weight
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.plasma, width=2)

# Step 7: Display the graph
plt.title("Energy Cost Between Nodes in Grid Network")
plt.show()