import numpy as np
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import csv
import re
import random as random

import os
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import heapq


csv_file = "arcs_51_51_fixed.csv"
initial_battery = 10000
"""
-------------------------------------------------------------------------------
Universals
-------------------------------------------------------------------------------
"""
speed_dic = {'charged':1, 'charging':1} #VALIDATE THESE SPEEDS THROUGH TESTING!
height_dic = {'charged':1.5, 'charging':1.5} #CONFIRM HEIGHTS - ASSUME 
seeker_orientation_uncertainty = {'human': (62*np.pi/180)+np.pi/2, 'bunker': 62*np.pi/180} #Enemy capabilities will influence this (among other things), will likely need to expand dictionary and definitions for each key
point_source_dic = {'charged': np.array([20/3, 10]), 'charging':np.array([40/3, 20])}
"""
-------------------------------------------------------------------------------
User Input
-------------------------------------------------------------------------------
"""
#Relative PATH in GitHub Project
file_path = ''
file_name=file_path+"Buckner" #map identifier
desired_lower_left_corner = (0, 0) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
desired_upper_right_corner = (500, 500) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
step_size=10  #desired distance, in meters, between nodes CANNOT BE TOO LARGE OR WILL CAUSE OverflowError when determining probability
# seekers ={1: [(5,5), 5, 0, seeker_orientation_uncertainty['human']]}
#seekers={1 : [(25,25), 5, 0, seeker_orientation_uncertainty['human']]} #, 2 : [(100,150), 15, -np.pi/2, seeker_orientation_uncertainty['human']], 3 : [(150,50), 10, 3*np.pi/4, seeker_orientation_uncertainty['bunker']]}
seekers={1 : [(25,25), 5, 0, seeker_orientation_uncertainty['human']], 
         2 : [(100,150), 15, -np.pi/2, seeker_orientation_uncertainty['human']], 
         3 : [(150,50), 10, 3*np.pi/4, seeker_orientation_uncertainty['bunker']], 
         4 : [(300,300), 10, 3*np.pi/4, seeker_orientation_uncertainty['bunker']],
         5 : [(400,100), 10, 11*np.pi/6, seeker_orientation_uncertainty['human']],
         6 : [(100,400), 10, 11*np.pi/6, seeker_orientation_uncertainty['human']]}

# seekers={1 : [(25,25), 5, 0, seeker_orientation_uncertainty['human']], 2 : [(50,50), 15, -np.pi/2, seeker_orientation_uncertainty['human']]}

# #{seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ], next seeker : [...], ...}
# fog_coef = 0

### DR Jane Vars

#Travel Time (s)
travel_time = 20

#Platform Name
platform_name = 'moose'

#Added Mass (kg)
added_mass = 513

#Wind Velocity (m/s)
wind_velocity = 0

#Originating Wind Direction (deg)
wind_direction = 0

generatorCoef = 100

"""
-------------------------------------------------------------------------------
Universal Calculations and Objects
-------------------------------------------------------------------------------
"""

elevation_map = Image.open(file_name+"_DEM.png") #read elevation map
vegetation_map = Image.open(file_name+"_NDVI.png") #read vegetation map
satelite_map = Image.open(file_name+".png")
max_elevation=603 #max elevation on the map (CAN WE READ THIS FROM A CSV or .txt?)
map_width_meters=3000 #how wide is the map in meters (CAN WE READ THIS FROM A CSV or .txt?) CHANGE TO 800 FOR desert
map_length_meters=2999 #how long is the map in meters (CAN WE READ THIS FROM A CSV or .txt?) CHANGE TO 500 FOR desert
desired_map_width = desired_upper_right_corner[0]-desired_lower_left_corner[0] #Determine the desired map width
desired_map_length = desired_upper_right_corner[1]-desired_lower_left_corner[1] #Determine the desired map length
left, top, right, bottom = [desired_lower_left_corner[0],map_length_meters-desired_upper_right_corner[1],desired_upper_right_corner[0],map_length_meters-desired_lower_left_corner[1]] #translate into image coordinates (flipped y-axis)
map_width_pixels, map_length_pixels = elevation_map.size #Get image origional size in pixels
vegetation_map = vegetation_map.resize((map_width_pixels, map_length_pixels)) #Resize vegetation map to elevation map (should only be off by a few pixels)
crop_width_scale = map_width_pixels/map_width_meters #Conversion for scaling map x-limits in meters to map corners in pixels
crop_length_scale = map_length_pixels/map_length_meters #Conversion for scaling map y-limits in meters to map corners in pixels
(left, top, right, bottom) = (left*crop_width_scale, top*crop_length_scale, right*crop_width_scale, bottom*crop_length_scale) #Convert requested map edges in meters to pixel edges

# return these
elevation_map = elevation_map.crop((left, top, right, bottom)) #grab correct elevation map
vegetation_map = vegetation_map.crop((left, top, right, bottom)) #grab correct vegetation map
map_width_pixels, map_length_pixels = elevation_map.size #get size in pixels
map_width_scale=(map_width_pixels-1)/desired_map_width #calculate scaling factor for converting x coordinate to pixel coordinate
map_length_scale=(map_length_pixels-1)/desired_map_length #calculate scaling factor for converting y coordinate to pixel coordinate
nodes_wide=int(desired_map_length/step_size)+1 #how many node columns
nodes_long=int(desired_map_length/step_size)+1 #how many node rows

def plot_contour(greens = False, levels = 10):
    '''
    Plots a contour map on either satelite imagery or vegetation map

    Parameters
    ----------
    greens : BOOL, optional
        True if want to show greens map.
        The default is False to show imagery.
    levels : INT, optional
        The amount of contour levels calculated. The default is 10.

    Returns
    -------
    None.

    '''
    plt.style.use('ggplot')
    fig, ax= plt.subplots() 
    xvals=np.linspace(0, desired_map_width,2*nodes_wide-1)
    yvals=np.linspace(0, desired_map_length,2*nodes_long-1)
    Z = []
    V = []
    k = 0

    for y in yvals:
        Z.append([])
        V.append([])
        for x in xvals:
            Z[k].append((elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation)
            V[k].append((vegetation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*3)
        k += 1
        
    Z = np.asarray(Z)
    V = np.asarray(V)
    
    X, Y = np.meshgrid(xvals, yvals)
    contours = plt.contour(X, Y, Z, levels, colors='black',)
    plt.clabel(contours, inline=True, fontsize=8)

    if greens == True:
        plt.imshow(V, extent=[0, xvals[-1], 0, yvals[-1]], cmap='Greens', alpha=0.5,  origin='lower')
        plt.colorbar(label='Vegetation')
        plt.title("Contour Map with Vegetation")
        
    if greens == False:
        ax.imshow(satelite_map, extent=[0, xvals[-1], 0, yvals[-1]], origin='lower', cmap='viridis') 
        plt.grid(False)
        plt.xlabel('X Offset')
        plt.ylabel('Y Offset')
        plt.title("Contour Map")

    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        ax.plot(xs,ys,color='red')
    
    return

def plot_sattelite():
    '''
    This function plots the image of a sattelite with the enemy locations on it

    Returns
    -------
    None.

    '''
    xvals=np.linspace(0, desired_map_width,2*nodes_wide-1)
    yvals=np.linspace(0, desired_map_length,2*nodes_long-1)
    plt.style.use('ggplot')
    fig, ax= plt.subplots()
    ax.imshow(satelite_map, extent=[0, xvals[-1], 0, yvals[-1]], origin='lower', cmap='viridis') 
    
    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        ax.plot(xs,ys,color='red')
    
    plt.title('Locations of Multiple Enemies')
    
    return


def get_node_id_master(node,w,h):
    """
    Given the master node number, return the node id in format 'm#'

    Parameters
    ----------
    node : master node number between 1 and 3*w*h
    w : INT. Width of the node field 
    h : INT. height of the node field


    """
    wh=w*h
    if node<=wh:
        node_pos='c'
    elif node<=2*wh:
        node_pos='s'
        node=node-wh
    elif node<=3*wh:
        node_pos='w'
        node=node-2*wh
    else:
        node_pos='r'
        node=node-3*wh
    return node_pos+str(node)
    
def get_node_id_vector(node_vec, w, h, node_pos, scale):
    """
    Given node location as a vector, and mode of movement, return the 
    node id in format 'm#'

    Parameters
    ----------
    node_vec : Vector_2D. x and y coordinates of the node
    w : INT. Width of the node field 
    h : INT. height of the node field
    node_pos : STR. either 'w','s', or 'c'
    scale : FLOAT. distance between nodes
    """
    node_num = int(((node_vec.x+node_vec.y+(w-1)*node_vec.y)/scale)+1)
    return node_pos+str(node_num)

def get_node_vector(node_id, w, h, scale):
    """
    Given node id, return x and y coordinates of the node as a vector

    Parameters
    ----------
    node_id : STR. Requested node in the format 'm#'
    w : INT. Width of the node field 
    h : INT. height of the node field
    scale : FLOAT. distance between nodes
    """
    node_num = int(node_id[1:])
    i = 1
    while i <= h:
        if node_num <= w*i:
            return np.array([node_num-w*(i-1)-1, i-1])*scale
        i += 1
    # print('SWW', node_id)
    return

def csv_to_dict(csv_file):
    data_dict = {}
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip header if present
        next(csv_reader, None)
        for row in csv_reader:
            key = int(row[0])  # Assuming the first column is the key
            values = int(row[1]) , int(row[2])  # Assuming the rest of the row are the values
            data_dict[key] = values
    return data_dict
  
def read_XK(start, end, file_name, w, h):
    """
    Reads 'output.csv' and returns the optimal path in terms of nodes

    Parameters
    ----------
    start : start node number (1-3*w*h)
    end : end node number (1-3*w*h)
    """
    with open(file_name, newline='') as opt:
        reader = csv.reader(opt)
        xk = list(reader)

    unsorted_arcs = np.transpose(np.asarray(xk[1:], dtype=float))[0]
    unsorted_path = []
    arc_dic = csv_to_dict(csv_file)
    # print(arc_dic)
    
    for a in unsorted_arcs:
        (sn, en) = arc_dic[a]
        unsorted_path.append([sn, en])

    visited = []
    PATH = [start]

    # print(unsorted_path)

    while PATH[-1] != end:
        for node_pair in unsorted_path:
            sn = node_pair[0]
            en = node_pair[1]
    

            if int(sn) == PATH[-1] and node_pair not in visited:
                #print("Current node pair:", node_pair)
                PATH.append(int(en))
                visited.append(node_pair)
                break
    return PATH

def create_title(file_name):
    match = re.match(r"output_batch_(\d+\.\d+)_(\d+\.\d+)\.csv", file_name)

    # Extract alpha and beta values
    alpha = match.group(1)
    beta = match.group(2)

    # Format the desired output
    return f"test - alpha: {alpha} beta: {beta}"


def plot_path(file_name, start, end, seekers=seekers, w=nodes_wide,h=nodes_long, scale=10, save=False):
    single_field = w*h 
    plot_contour()
    path = read_XK(start, end, file_name, w, h)

    print('Paths Followed:',path)

    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        
    "Plot Path"
    pscale = {'w': 0, 'c': 2, 's': 1}
    pathx = []
    pathy = []
    px = [[], [], []]
    py = [[], [], []]

    for node in path:
        node_id = get_node_id_master(node, w, h)
        node_vec = get_node_vector(node_id, w, h, scale)
        px[pscale[node_id[0]]].append(node_vec[0])
        py[pscale[node_id[0]]].append(node_vec[1])
        pathx.append(node_vec[0])
        pathy.append(node_vec[1])

    # Plot path


    for i in range(len(pathx) - 1):
    # Check if the segment is in the special set
        segment = (path[i], path[i + 1])
        # print(path[i])
        if (path[i] > single_field) or (path[i]+1 > single_field):
            color = 'green'
        else:
            color = 'black'
        
    # Line Segments
        plt.plot([pathx[i], pathx[i + 1]], [pathy[i], pathy[i + 1]], color=color, linewidth=3)

    # Create Title

    output_string = create_title(file_name)
        
        
    plt.plot([], [], color='green', label='charging')  # Placeholder for legend
    plt.plot([], [], color='black', label='charged')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(output_string)
    plt.grid(True)
    plt.legend()
    plt.show()

    if save:
        plt.savefig(f"{output_string}.png", dpi=300)  # Save as PNG with high resolution (300 dpi)

def plot_all(listOfPaths, start,end, detection=False,w=nodes_wide, h=nodes_long, scale=10):
    single_field = w*h 
    plot_contour()

    labels = []
    for file_name in listOfPaths:
        labels.append(create_title(file_name))
    
    colors = ['black']
    colors = ["#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(listOfPaths))]


    addedLabels = []

    for idx, path in enumerate(listOfPaths):

        path = read_XK(start, end, path, w, h)
        
        for seeker in seekers:
            [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
            # ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
            thetas=np.linspace(0, 2*np.pi,100)
            xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
            ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
            # ax.plot(xs,ys,color='red')
            
        "Plot Path"
        pscale = {'w': 0, 'c': 2, 's': 1}
        pathx = []
        pathy = []
        px = [[], [], []]
        py = [[], [], []]

        for node in path:
            node_id = get_node_id_master(node, w, h)
            node_vec = get_node_vector(node_id, w, h, scale)
            px[pscale[node_id[0]]].append(node_vec[0])
            py[pscale[node_id[0]]].append(node_vec[1])
            pathx.append(node_vec[0])
            pathy.append(node_vec[1])

        # Plot path
        for i in range(len(pathx) - 1):
            segment = (path[i], path[i + 1])
            plt.plot([pathx[i], pathx[i + 1]], [pathy[i], pathy[i + 1]], 
                 linewidth=3, color=colors[idx], alpha=1)

        if labels[idx] not in addedLabels:
            plt.plot([], [], color=colors[idx], label=labels[idx])
            addedLabels.append(labels[idx])

    plt.xlabel('X Offset')
    plt.ylabel('Y Offset')
    plt.title('Energy')
    plt.grid(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
    plt.show()

def main(start, end):
    folder_path = "."  # Replace with the path to your folder
    csv_files = [f for f in os.listdir(folder_path) 
                if f.endswith('.csv') 
                and os.path.isfile(os.path.join(folder_path, f)) 
                and 'output' in f.lower()]
    
    # Print each CSV file name
    for file in csv_files:
        print(file)
   
    # with all of the paths that is in the folder
    # for file in csv_files:
    #     plot_path(csv_files, start, end, seekers=seekers, w=nodes_wide,h=nodes_long, scale=10, save=False)
    
    plot_all(csv_files, start,end, detection=False,w=nodes_wide, h=nodes_long, scale=10)

if __name__ == "__main__":
    start = 1
    end = 2601

    main(start, end)