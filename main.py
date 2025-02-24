import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import csv
from tqdm import tqdm

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from exenf_alog import exenf_cost

from detection_funcs import get_audio_detection, seeker_orientation_uncertainty, get_seeker_group, get_visual_detection

# Constants 
SITUATION = "Buckner"
SPACING = 10
MAX_ELEVATION = 603
DISTANCE_SCALE = 45
GENERATOR_COEF = 5 # J per Second
SPEED = 6.7 # m/s
height_dic = {'charged':1.5, 'charging':1.5} #CONFIRM HEIGHTS - ASSUME 

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


# Seekers 
seekers={1 : [(60,60), 5, 0, seeker_orientation_uncertainty['human']],
         2 : [(30,30), 5, np.pi/2, seeker_orientation_uncertainty['human']],
         3 : [(90,90), 5, np.pi/2, seeker_orientation_uncertainty['human']]}

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
    
    def calculate_arc(self, neighbor, seeker_groups):
        """Calculate arc properties between this node and a neighbor."""
        platform_name = 'moose'
        added_mass = 0
        wind_velocity = 0
        wind_direction = 0

        position_self = np.array([self.x, self.y, self.e])
        position_neighbor = np.array([neighbor.x, neighbor.y, neighbor.e])

        travel_time = SPACING * DISTANCE_SCALE / SPEED

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
        
        visual_detection = get_visual_detection(position_self,position_neighbor, mode_of_travel, travel_time, seeker_groups, seekers, elevation_map, MAX_ELEVATION,vegetation_map,DISTANCE_SCALE)
        audio_detection = get_audio_detection(position_self,position_neighbor,mode_of_travel,seeker_groups, vegetation_map,DISTANCE_SCALE)        
        risk_level = max(visual_detection, audio_detection)

        # risk_level = 100

        # Energy cost
        if np.array_equal(position_self, position_neighbor):
            energy_cost = 0
        else: 
            heading = direction_of_travel(position_self,position_neighbor)
            if heading == None:
                heading = 1
        
            params = [position_self, position_neighbor, travel_time, platform_name, added_mass, wind_velocity, wind_direction, heading, False]
            fcns = [np, minimize, interp1d, math, os]
            Jcon, Jgen, msg = exenf_cost(params,fcns)
            # Jgen = Jgen*(.25)

            energy_cost = Jcon - Jgen
            
            if mode_of_travel == 'charging':
                Jgen += GENERATOR_COEF * travel_time
                energy_cost = Jcon - Jgen

        return risk_level, travel_time, energy_cost, movement_code

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
    """creates 3D grid of nodes with all attributes neccesary for cost function
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
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker], elevation_map, MAX_ELEVATION) for templated_seeker in seekers}

    total_iterations = sum(
        len(row) for z in range(len(grid)) for row in grid[z]
    )

    with tqdm(total=total_iterations, desc="Processing nodes") as pbar:
        for z in range(len(grid)):  # Iterate through top and bottom grids
            for row in grid[z]:
                for node in row:
                    neighbors = node.find_neighbors(grid, rows, cols)
                    for neighbor in neighbors:
                        risk_level, travel_time, energy_cost, movement_code = node.calculate_arc(neighbor, seeker_groups)
                        arc_dict[arc_id] = node.id, neighbor.id, risk_level, travel_time, energy_cost, movement_code
                        arc_id += 1
                    pbar.update(1)  # Update progress for each node processed
    
    return arc_dict


def direction_of_travel(initial_point, final_point):
    x1, y1, _ = initial_point
    x2, y2, _ = final_point
    
    dx = x2 - x1
    dy = y2 - y1

    heading = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    return heading


def display_grid(super_grid, img=None):
    """_summary_

    Args:
        super_grid (numpy.array): _description_
        img (image name, optional): _description_. Defaults to None.
    """
    fig, ax= plt.subplots()
    step_size = 5

    if img is not None:
        plt.imshow(img)
    
    for grid in super_grid:
        for row in grid:
            for node in row:
                plt.scatter(node.x, node.y, color="black", s=5)

    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        ax.plot(xs,ys,color='red')
    
    plt.show()

def get_in_out_arcs(arc_dictionary):
    """
    Create dictionaries for all arcs flowing into and out of a node.

    :param arc_dictionary: Dictionary of arcs with keys as arc identifiers and values as [start_node, end_node, risk, time, energy_level, movement_code]
    :return: Two dictionaries:
        - in_arcs: Arcs flowing into each node.
        - out_arcs: Arcs flowing out of each node.
    """
    in_arcs = {}
    out_arcs = {}

    for arc, arc_info in arc_dictionary.items():
        start_node, end_node, risk, time, energy_level, movement_code = arc_info

        # Add arc to out_arcs for the start_node
        if start_node not in out_arcs:
            out_arcs[start_node] = []
        out_arcs[start_node].append(arc)

        # Add arc to in_arcs for the end_node
        if end_node not in in_arcs:
            in_arcs[end_node] = []
        in_arcs[end_node].append(arc)

    return in_arcs, out_arcs

from collections import defaultdict

def get_triangle_sets(super_grid, arc_dictionary):
    pass


def pad_to_length(data, length):
    if len(data) < length:
        data.extend([0] * (length - len(data)))
    return data

def write_to_csv(situation_name, super_grid, arc_dictionary):
    arc_name = situation_name + f"_arcs_{rows}_{cols}.csv"
    ins_name = situation_name + f"_ins_{rows}_{cols}.csv"
    outs_name = situation_name + f"_outs_{rows}_{cols}.csv"
    triangles_name = situation_name + f"_triangles_{rows}_{cols}.csv"

    # header = ['Arc', 'Start Node', 'End Node', 'Risk', 'Time', 'Energy Level', 'Movement Code']

    with open(arc_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # writer.writerow(header)
        for arc in arc_dictionary:
            [start_node, end_node, risk, time, energy_level, movement_code] = arc_dictionary[arc] 
            writer.writerow([arc, start_node, end_node, risk, time, energy_level, movement_code])

    "Get inflow sets, outflow sets, and triangle sets"
    in_arcs, out_arcs = get_in_out_arcs(arc_dictionary)

    'Write outflow dictionary to csv'
    with open(outs_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for node in range(N):
            outflows = out_arcs[node+1]
            outflows_padded = pad_to_length(outflows, 10)
            writer.writerow(outflows_padded)

    'Write inflow dictionary to csv'
    with open(ins_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for node in range(N):
            inflows = in_arcs[node+1]
            inflows_padded = pad_to_length(inflows, 10)
            writer.writerow(inflows_padded)

    # triangle_sets = get_triangle_sets(super_grid, arc_dictionary)

    # with open(triangles_name, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for triangle in triangle_sets:
    #        writer.writerow(triangle)       


def extract_arcs(csv_name):
    arcs = {}
    energy_cost = []
    with open(csv_name, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            key = int(row[0])  # First column as key
            value = (int(row[1]), int(row[2]))  # Tuple of second and third columns
            energy_cost.append(float(row[5]))
            arcs[key] = value
    return arcs,energy_cost

def extract_path(csv_name):
    path = []
    with open(csv_name, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            path.append(float(row[0]))
    return path
    
def plot_path(arcs, path, super_grid, img):
    fig, ax = plt.subplots()
    step_size = 5

    # Plot image if provided
    if img is not None:
        ax.imshow(img)
    
    # Plot Nodes
    for grid in super_grid:
        for row in grid:
            for node in row:
                ax.scatter(node.x, node.y, color="black", s=5)


    for seeker in seekers:
        (seeker_x, seeker_y), z, seeker_orient, _ = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2 * step_size * np.cos(seeker_orient), 2 * step_size * np.sin(seeker_orient),
                 width=step_size / 10, head_width=step_size / 2, color='red')

        thetas = np.linspace(0, 2 * np.pi, 100)
        xs = [seeker_x + z * np.cos(theta) for theta in thetas]
        ys = [seeker_y + z * np.sin(theta) for theta in thetas]
        ax.plot(xs, ys, color='red')

    # Path Code 

    nodes = [coord for arc in path for coord in arcs[arc]]
    x_coords, y_coords, z = zip(*[find_node_by_id(node, super_grid) for node in nodes])

    x_coords = [x - 144 if x > 144 else x for x in x_coords]
    y_coords = [y - 144 if y > 144 else y for y in y_coords]

    for i in range(len(x_coords) - 1):

        color = 'green' if z[i] == 1 else 'blue'  # Use red for z=1, blue otherwise

        ax.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color=color, linewidth=3)

    plt.show()


def find_node_by_id(node_id, super_grid):
    for z in range(len(super_grid)): 
        for row in super_grid[z]:
            for node in row:  
                if node.id == node_id: 
                    return (node.x, node.y, z)
    return None  # Return None if node ID is not found

def plot_battery(path, energy_cost):
    energy_for_path = []
    for p in path:
        energy_for_path.append(energy_cost[int(p-1)])

    plt.bar(range(len(energy_for_path)), energy_for_path)
    plt.xlabel("Step")
    plt.ylabel("Cost Per Step")
    plt.title("Battery Cost Per Step")
    plt.show()
    print(sum(energy_for_path))

    return (energy_for_path)

def plot_energy(energy_cost):
    plt.bar(range(len(energy_cost)), energy_cost)
    plt.show()
    print(sum(energy_cost))
    return (energy_cost)


def plot_battery_depletion(battery_capacity, energy_cost):
    battery_levels = [battery_capacity]  # Start with full battery
    
    energy_for_path = []
    for p in path:
        energy_for_path.append(energy_cost[int(p-1)])

    for energy in energy_for_path:
        battery_levels.append(battery_levels[-1] - energy)

    steps = list(range(len(battery_levels)))

    plt.plot(steps, battery_levels, marker='o', linestyle='-', color='b', label="Battery Level")
    plt.axhline(y=0, color='r', linestyle='--', label="Empty Battery")
    
    plt.xlabel("Step")
    plt.ylabel("Battery Level")
    plt.title("Battery Depletion Over Step")
    plt.legend()
    plt.grid(True)
    
    plt.show()

def order_path(arcs, path):
    result = {}
    for p in path:
        result[p] = arcs[p]

    ordered_arcs = []
    
    arc_items = list(result.items())
    
    start_arc = arc_items[0]
    ordered_arcs.append(start_arc)
    arc_items.remove(start_arc)

    while arc_items:
        # Find the next arc that starts where the last one ends
        for arc in arc_items:
            if ordered_arcs[-1][1][1] == arc[1][0]:  # Check if the end of the last arc matches the start of the current arc
                ordered_arcs.append(arc)
                arc_items.remove(arc)  # Remove the arc once it's added
                break
    
    # Return the ordered arcs (just the values)
    ordered_arcs_with_values = [arc[0] for arc in ordered_arcs]
    print(ordered_arcs_with_values)
    return ordered_arcs_with_values


        

super_grid = setup(rows, cols)
plot = True


if plot == True: 
    path = extract_path('output_12x12.csv')
    arcs,energy_cost = extract_arcs('Buckner_arcs_12_12.csv')
    # print(energy_cost)
    print(plot_battery(path, energy_cost))
    plot_battery_depletion(500, energy_cost)
    path = order_path(arcs, path)
    plot_path(arcs, path, super_grid, img=sat_map)
else: 
    arc_dictionary = get_arcs(super_grid)
    display_grid(super_grid, sat_map)
    write_to_csv(SITUATION, super_grid, arc_dictionary)

