import numpy as np
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import csv
import random as random

import os
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from exenf_alog import direction_of_travel
from exenf_alog import exenf_cost

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


chance_snow = 0
chance_rain = 0 
fog = 0 

background_noise = 50
    
def detection_fields(mode_of_travel, perpendicular=True, plot_node_field=False, seekers=seekers, path=False):
    '''
    Plots detection fields of given enemies

    Parameters
    ----------
    mode_of_travel : STR
        'walking', 'sneaking', or 'crawling'
    perpendicular : BOOL
        Is seeker moving perpendicular to seekers direct line of sight.
    plot_node_field : BOOL
        Plot nodes over given map.
    seekers : DICT, optional
        Dictionary of seekers to plot. The default is seekers.

    Returns
    -------
    None.

    '''
    aud_lis = []
    
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker]) for templated_seeker in seekers}
    xvals=np.linspace(0, desired_map_width,2*nodes_wide-1)
    yvals=np.linspace(0, desired_map_length,2*nodes_long-1)
    travel_time = int(step_size/speed_dic[mode_of_travel])
    detection=[]
    # checked_locations=0
    for y in tqdm(yvals, desc="Progress", position=0, leave=True):
        detection.append([])
        for x in xvals:
            
            elevation= (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation
            position_i=np.array([x,y, elevation])
            visual_detection = get_visual_detection_2(position_i, mode_of_travel, travel_time, seeker_groups, perpendicular)
            #visual_detection = 0 
            audio_detection = get_audio_detection(position_i, position_i, mode_of_travel, seeker_groups)*2 #WILL NEED TO FIX
            #aud_lis.append(audio_detection)
            #audio_detection = 0
            # checked_locations+=1
            # print(progress_bar(np.round(checked_locations/total_locations, 2)))
            detection[-1].append(max(visual_detection,audio_detection))
    plt.clf()
    plt.style.use('ggplot')
    fig, ax= plt.subplots()
    im = ax.imshow(detection, extent=[0, xvals[-1], 0, yvals[-1]],vmin=0,vmax=1,
                    origin='lower', cmap='viridis')       

    # cb_ax = fig.add_axes([0.83, 0.3, 0.02, 0.4])
    # fig.colorbar(im, cax=cb_ax, label="Detection Probability")
    
    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        ax.plot(xs,ys,color='red')
    fig.colorbar(im, label="Detection Probability")
    if plot_node_field:
        node_field=create_node_field_robo()
        for i in range(nodes_wide*nodes_long):
            (x,y)=node_field[i+1][0]
            plt.scatter(x,y,color="black")
    plt.title('Detection Radius for Multiple Enemies', fontsize=20)
    
    if path:
        situtation_name = 'test' 
        start = 1
        end = 441
        seekers=seekers
        w=nodes_wide
        h=nodes_long
        scale=10
        plt.style.use('ggplot')
        fig, ax= plt.subplots() 

        plot_contour()

        path = read_XK(start, end, "output_"+situtation_name +".csv", w, h)
        print('here')

            
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
        s = 100
        plt.plot(pathx, pathy, color='black', linewidth=3)
        plt.scatter(px[0], py[0], color='green', label="Walking", s=s)
    
    
    
    return     

    
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

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
"""
---------------------------------ROBO-------------------------------------------
"""
def get_arcs_robo(nodes_wide=nodes_wide, nodes_long=nodes_long, step_size=step_size, file_name=file_name, max_elevation=max_elevation, seekers=seekers):
    """ 
    gets arcs and node field

    Parameters
    ----------
    nodes_wide : Integer
        width of node field measured in nodes.
    nodes_long : Integer
        length of node field measured in nodes.
    step_size : Float
        desired birds eye distance between N-S or E-W adjacent nodes.
    file_name : String
        Name of map: typically changed universally as 'map_name_location'
    max_elevation : Float
        Maximum elevation on elevation imagery.
    seekers : dictionary
        { seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ]}. NEED TO ADD CAPABILITIES

    Returns
    -------
    arcs : Dictionary
        Keys are tuples that indicate the arcs start and end nodes e.g. (i,j).
        definitions are node coordinates (including elevation), mode of travel, travel time, and probabiliy of detection (NEED TO ADD FUEL CONSUMPTION)
        e.g. {(node i, node j) : [ node_i location (x,y,z), node_j location (x,y,z), mode of travel, time, detection probability ], ... }
    node_field : dictionary
        Keys are node id numbers (1,...,3wl) and definitions are coordinates (as a tuple), elevation, and adjacent nodes (as a list)
        e.g. {Node Id Number : (x,y), elevation, vegetation, [adjacent nodes], ... }.
    """
    # start_time = time.time()
    
    node_field=create_node_field_robo()
    
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker]) for templated_seeker in seekers}
    
    single_field=nodes_wide*nodes_long
    
    "Create arcs dictionary"
    arcs={}
    # arc_length=get_arc_length()/2

    charging_nodes=[i+1+single_field for i in range(single_field)]
    
    checked=0

    for i in tqdm(range(0,2*single_field)):
        node_i=i+1

        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        position_i=np.array(coordinate_i+(elevation_i,))
        for node_j in adjacent_nodes_i:
            "Dont recaclulate arcs arleady found"
            key=(node_j,node_i)
            if key not in arcs:
                debug = False
                
                [coordinate_j, elevation_j, vegetation_j, adjacent_nodes_j] = node_field[node_j]
                position_j=np.array(coordinate_j+(elevation_j,))
                
                "Determine Time for travel"
                distance = np.linalg.norm(position_j-position_i)
                
                
                
                "Determine time base on speed, i average vegetation encountered"
                average_vegetation=(vegetation_i+vegetation_j)/2
                vegetation_factor=1-(2/9)*average_vegetation #NEEDS VALIDATION THROUGH TESTING!!!!!!!!!!!
                
                

                if coordinate_i == coordinate_j:
                    mode_of_travel = 'charging'
                    travel_time = 60
                if node_i in charging_nodes:
                    mode_of_travel='charging'
                    travel_time=distance/(vegetation_factor*speed_dic[mode_of_travel])
                else:
                    mode_of_travel='charged'
                    travel_time=distance/(vegetation_factor*speed_dic[mode_of_travel])
                "Determine Probability of detection"
                visual_detection=get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups)
                audio_detection=get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups)

                risk_level = max(visual_detection,audio_detection)    
                "Determine the Energy Cost"

                p1 = [position_i[0] , position_i[1] , position_i[2]]
                p2 = [position_j[0] , position_j[1] , position_j[2]]

                if p1 == p2:
                    energy_cost = 0

                else:
                    #Input Parameters
                    heading = direction_of_travel(p1,p2,math)
                    if heading == None:
                        heading = 1
                
                    params = [p1, p2, travel_time, platform_name, added_mass, wind_velocity, wind_direction, heading, debug]

                    #Input Function Handles
                    fcns = [np, minimize, interp1d, math, os]

                    #Exergy/Energy Cost
                    Jcon, Jgen, msg = exenf_cost(params,fcns)
                    #energy_level = 1

                    # Apply a penalty for regenerative braking 
                    Jgen = Jgen*(.25)
                    
                    if mode_of_travel == 'charging':
                        Jgen += 50

                    energy_cost = Jcon - Jgen
                #TEST CODE
                # energy_cost = 0
                # risk_level = 0
                # print(energy_cost)
                arcs[(node_i,node_j)]=[position_i, position_j, mode_of_travel, travel_time, risk_level, energy_cost, elevation_i]
                arcs[(node_j,node_i)]=[position_j, position_i, mode_of_travel, travel_time, risk_level, energy_cost, elevation_i]
                checked+=1

    return arcs, node_field

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


def create_node_field_robo():
    """
    Creates node field

    Parameters (All Universal)
    ----------
    nodes_wide : Integer
        width of node field measured in nodes.
    nodes_long : Integer
        length of node field measured in nodes.
    step_size : Float
        desired birds eye distance between N-S or E-W adjacent nodes.
    file_name : String
        Name of map: typically changed universally as 'map_name_location'
    max_elevation : Float
        Maximum elevation on elevation imagery.

    Returns
    -------
    node_field : Dictionary
        Keys are node id numbers (1,...,3wl) and definitions are coordinates (as a tuple), elevation, and adjacent nodes (as a list)
        e.g. {Node Id Number : (x,y), elevation, vegetation, [adjacent nodes], ... }.

    """
    node_field={}
    single_field=nodes_wide*nodes_long


    node_id=1
    for l in range(nodes_long):
        for w in range(nodes_wide):
            coordinate = (w*step_size,l*step_size)
            (x,y)=(coordinate[0]*map_width_scale,coordinate[1]*map_length_scale)
            elevation = (elevation_map.getpixel((x, y))[0]/255)*max_elevation
            r= vegetation_map.getpixel((x, y))[0] 
            vegetation = (3-(3*(r/255))) #vegetation is scaled "continuously" from 0 (None) to 3 (Dense)
            #charged level
            node_field[node_id] = [coordinate, elevation+1, vegetation, get_adjacent_nodes(node_id, coordinate)]
            #charging level
            node_field[node_id+single_field] = [coordinate, elevation+1, vegetation, get_adjacent_nodes(node_id+single_field, coordinate)]
            node_id+=1            
    return node_field

        
def get_adjacent_nodes(node_id, coordinate):
    """
    Gets the adjacent nodes for a specific node number

    Parameters
    ----------
    node_id : Integer
        The node which you wish to know its adjacent nodes.
    coordinate : tuple
        x and y coordinates of node_id.
    nodes_wide : Integer (UNIVERSAL)
        width of node field measured in nodes.
    nodes_long : Integer (UNIVERSAL)
        length of node field measured in nodes.
    step_size : Float (UNIVERSAL)
        desired birds eye distance between N-S or E-W adjacent nodes.

    Returns
    -------
    actual_adjacents : list
        list of node ID numbers that are adjacent to the given node_id

    """
    single_field=nodes_wide*nodes_long
    potential_adjacents=[node_id+1,node_id+nodes_wide+1,node_id+nodes_wide,
                         node_id+nodes_wide-1,node_id-1,node_id-nodes_wide-1, 
                         node_id-nodes_wide,node_id-nodes_wide+1, 
                         node_id+single_field,node_id-single_field] #Possible nodes starting at theta=0 and proceeding pi/4 and then adding next level and previous level
    
    potential_adjacents_locations=[(coordinate[0]+step_size,coordinate[1]),(coordinate[0]+step_size,coordinate[1]+step_size),(coordinate[0],coordinate[1]+step_size),
                                   (coordinate[0]-step_size,coordinate[1]+step_size),(coordinate[0]-step_size,coordinate[1]),(coordinate[0]-step_size,coordinate[1]-step_size),
                                   (coordinate[0],coordinate[1]-step_size),(coordinate[0]+step_size,coordinate[1]-step_size),coordinate,coordinate] #Possible node locations starting at theta=0 and proceeding pi/4 and then adding next level and previous level
    actual_adjacents=[]

    for i in range(10):
        potential_location=potential_adjacents_locations[i]
        in_horizon = potential_adjacents[i]>0 and potential_adjacents[i]<=2*single_field
        in_map = potential_location[0]>=0 and potential_location[0]<step_size*nodes_wide and potential_location[1]>=0 and potential_location[1]<step_size*nodes_long
        if in_horizon and in_map:
            actual_adjacents.append(potential_adjacents[i])

            
    return actual_adjacents

def classify_node(position_i):
    pass
    
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
        plt.title("Contour Map with Satelite Imagery")

    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        ax.plot(xs,ys,color='red')
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    return


"""
-------------------------------------------------------------------------------
Generate Outputs
-------------------------------------------------------------------------------
""" 

def write_csv_files(scenario_name, check=False):
    """
    Write csv files of scenario as used in the julia optimization code.

    Parameters
    ----------
    scenario_name : string
        what you would like the scenario to be called.

    Returns
    -------
    None.
    
    Writes
    -------
    arc_name.csv : .csv file
        Each row is [arc ID, node_i, node_j, risk, time, movement_code] where
        movement code is 0, 1, 2, or 3 for movement transition, crawling, 
        sneaking, and walking, respectively.
    
    group_name.csv : .csv file (Only generated if needed)
        Each column is a list of arc IDs that are crawling arcs (column 1) or 
        sneaking arcs (column 2)
    
    tri_name.csv : .csv file (Only generated if needed)
        Each row correspond to a set of arc IDs that form a triangle
    
    in_name.csv : .csv file (Only generated if needed)
        row i is the list of arc IDs flowing into node i
        
    out_name.csv : .csv file (Only generated if needed)
        row i is the list of arc IDs flowing out of node i

    """
    # Get repeatedly used dictionaries, strings, and values
    
    middle_name = str(nodes_wide) + '_' + str(nodes_long)
    # print(middle_name)
    arcs, node_field = get_arcs_robo()
    N = len(node_field)
    # N = 2 * (nodes_wide*nodes_long)

    ordered_arcs = get_ordered_arcs(arcs, node_field, N)
    
    'Write Arcs file to csv'
    # crawling_arcs = []
    # sneaking_arcs = []
    arc_name = 'arcs_' + middle_name + '_' + scenario_name + '.csv'
    with open(arc_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for arc in ordered_arcs:
            (node_i, node_j) = arc
            [arc_ID, risk, time, energy_level, movement_code] = ordered_arcs[arc] 
            writer.writerow([arc_ID, node_i, node_j, risk, time, energy_level, movement_code])

    "Write Node Field Structure csv files, if needed"
    tri_name = middle_name + '_triangle.csv'
    in_name = middle_name + '_inflow.csv'
    out_name = middle_name + '_outflow.csv'
    
    if path.exists(tri_name) and path.exists(out_name):
        print("Inflows, outflows, and triangle sets skipped: already written")
        return 
    
    "Get inflow sets, outflow sets, and triangle sets"
    node_i_inflows, node_i_outflows = get_ins_outs(ordered_arcs,N)
    triangle_sets = get_triangle_sets(node_field, ordered_arcs)
    
    'write triangle relationships to csv'
    with open(tri_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(triangle_sets)
    
    'Write outflow dictionary to csv'
    with open(out_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for node in range(N):
            outflows = node_i_outflows[node+1]
            outflows_padded = pad_to_length(outflows, 10)
            writer.writerow(outflows_padded)

    'Write inflow dictionary to csv'
    with open(in_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for node in range(N):
            inflows = node_i_inflows[node+1]
            inflows_padded = pad_to_length(inflows, 10)
            writer.writerow(inflows_padded)
                
    return

def get_ordered_arcs(arcs, node_field, N):
    """
    Takes the arcs dictionary and sorts it, giving each arc a unique arc ID.

    Parameters
    ----------
    arcs : Dictionary
        Keys are tuples that indicate the arcs start and end nodes e.g. (i,j).
        definitions are node coordinates (including elevation), mode of travel, travel time, and probabiliy of detection (NEED TO ADD FUEL CONSUMPTION)
        e.g. {(node i, node j) : [ node_i location (x,y,z), node_j location (x,y,z), mode of travel, time, detection probability ], ... }
    node_field : Dictionary
        Keys are node id numbers (1,...,3wl) and definitions are coordinates (as a tuple), elevation, and adjacent nodes (as a list)
        e.g. {Node Id Number : (x,y), elevation, vegetation, [adjacent nodes], ... }.

    Returns
    -------
    ordered_arcs : Dictionary
        Keys are tuples that indicate the arcs start and end nodes e.g. (i,j).
        definitions are arc Id numbers, probability of detection, travel time and a movement code
        e.g. {(node i, node j) : [arc_ID, node_i, node_j, risk, time, movement_code ], ... }
    """
    travel_dic = {'charged':0, 'charging':1} #Used to produce the movement code
    arc_ID = 1 #Initiate arc IDs
    ordered_arcs={} # Initiate ordered arcs dictionary

    for i in range(N): 
        node_i = i + 1
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        for node_j in adjacent_nodes_i:
            [ (x_i, y_i, z_i), (x_j, y_j, z_j), mode_of_travel, time, risk, energy_level ] = arcs[(node_i, node_j)]
            ordered_arcs[(node_i, node_j)] = [arc_ID, risk, time, energy_level, travel_dic[mode_of_travel]]
            arc_ID += 1
    return ordered_arcs

def get_triangle_sets(node_field, ordered_arcs):
    """
    Get triangle relationships for a specific node_field with designated arc IDs
    Commented out sections were to determine if any repeated triangles were found (they were not)

    Parameters
    ----------
    node_field : Dictionary
        Keys are node id numbers (1,...,3wl) and definitions are coordinates (as a tuple), elevation, and adjacent nodes (as a list)
        e.g. {Node Id Number : (x,y), elevation, vegetation, [adjacent nodes], ... }.
    ordered_arcs : Dictionary
        Keys are tuples that indicate the arcs start and end nodes e.g. (i,j).
        definitions are arc Id numbers, probability of detection, travel time and a movement code
        e.g. {(node i, node j) : [arc_ID, node_i, node_j, risk, time, movement_code ], ... }

    Returns
    -------
    triangle_sets : list of lists
        each list in this list represents a set of arc IDs that form a triangle step.
        For example, if set1 = [a_ij, a_jk, a_ik] then only one of these arc IDs should ever be used.
        That is going from i to j (a_ij) and then from j to k (a_jk) is the same as going from
        i to k (a_ik).

    """
    # triangle_sets=[np.array([0,0,0])]
    triangle_sets=[]
    for node_i in node_field:
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        for node_j in adjacent_nodes_i:
            [coordinate_j, elevation_j, vegetation_j, adjacent_nodes_j] = node_field[node_j]
            for node_k in adjacent_nodes_j:
                if node_k in adjacent_nodes_i:
                    [arc_ij, risk, time, energy_level, movement_code] = ordered_arcs[(node_i, node_j)] 
                    [arc_jk, risk, time, energy_level, movement_code] = ordered_arcs[(node_j, node_k)] 
                    [arc_ik, risk, time, energy_level, movement_code] = ordered_arcs[(node_i, node_k)] 
                    triangle_set = [arc_ij, arc_jk, arc_ik]
                    # triangle_set = np.sort([arc_ij, arc_jk, arc_ik])
                    triangle_sets.append(triangle_set)
                    # print(triangle_set, triangle_sets)
                    # if not np.any(np.all(triangle_set == np.array(triangle_sets), axis = 1)):
                    #     triangle_sets.append(triangle_set)
                    # else:
                    #     print("woah")
    
    return triangle_sets


def get_ins_outs(ordered_arcs, N):
    """
    Get sets of arc ID numbers that flow into and out of all nodes

    Parameters
    ----------
    ordered_arcs : dict
        arcs rearranged to be in order (all arcs coming out of node 1, then node 2, etc)
        Given an arc ID. {(node_i, node_j) : [arc_ID, risk, time, movement_code], etc...}
    N : Int
        number of nodes in the nodefield.

    Returns
    -------
    node_i_inflows : dictionary
        sets of arc_IDs flowing into node i {node_i : [all arc_IDs that flow into node_i], etc...}.
    node_i_outflows : dictionary
        sets of arc_IDs flowing out of node i {node_i : [all arc_IDs that flow out of node_i], etc...}.

    """
    node_i_inflows = {i+1 : [] for i in range(N)}
    node_i_outflows = {i+1 : [] for i in range(N)}
    for outflow_arc in ordered_arcs:
        (node_i, node_j) = outflow_arc
        inflow_arc = (node_j, node_i)
        [outflow_arc_ID, risk, time, energy_level, movement_code] = ordered_arcs[outflow_arc] 
        [inflow_arc_ID, risk, time, energy_level, movement_code] = ordered_arcs[inflow_arc] 
        node_i_inflows[node_i].append(inflow_arc_ID)
        node_i_outflows[node_i].append(outflow_arc_ID)
        
    return node_i_inflows, node_i_outflows

def pad_to_length(data, length):
    if len(data) < length:
        data.extend([0] * (length - len(data)))
    return data

def get_visual_detection_2(position_i, mode_of_travel, travel_time, seeker_groups, perpendicular):

    for seeker in seekers:
        [seeker_coord, z, orient, orient_uncert] = seekers[seeker]
        distance_position_i=np.linalg.norm(np.array(seeker_coord)-position_i[:2])
        if distance_position_i <=z*np.sqrt(2):
            return 1
    visual_detection=[]
    for worst_case_seekers in seeker_groups:

        for seeker in seeker_groups[worst_case_seekers]:
            los = get_los(seeker, position_i) #Line of Sight to start postion
            [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
            seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
            seeker_to_evader = position_i-seeker_coord            
            #print(seeker_to_evader)
            if los==0:
                visual_detection.append(0)
            #elif seeker_to_evader > fog_coef.all():
            #    visual_detection.append(0)
            else:
                [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
                seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
                seeker_to_evader = position_i-seeker_coord
                if perpendicular:
                    evader_v = rotate(np.pi/2) @ seeker_to_evader[:2]
                    evader_v = evader_v / np.linalg.norm(evader_v) #move perpendicular to seekers direct line of sight
                    position_j = position_i + speed_dic[mode_of_travel]*np.append(evader_v, position_i[-1])
                else:
                    evader_v = seeker_to_evader[:2] / np.linalg.norm(seeker_to_evader[:2])
                    position_j = position_i + speed_dic[mode_of_travel]*np.append(evader_v, position_i[-1])
                alpha = get_alpha(seeker_coord, position_i, position_j, speed_dic[mode_of_travel])
                beta = get_beta(seeker_coord, position_i, position_j, height_dic[mode_of_travel])
                trace_ratio = closed_form_ratio(alpha, beta)
                # detection_probability = 999*trace_ratio/(998*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = 101*trace_ratio/(100*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = detection_over_step(int(travel_time), detection_probability) #testing this function out
                visual_detection.append(los*detection_probability)
    return max(visual_detection)


def get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups):
    '''
    This function should get the probability of visual detection
    Tasks:
        o Determine line of sight to start and end positions (seperate function that checks if evader is in deadspace or is blocked by foliage)
        o Do the necessary calculus (Jakes function)

    Parameters
    ----------
    position_i : LIST
        Node i [xcoord, ycoord, elevation].
    position_j : LIST
        Node j [xcoord, ycoord, elevation].
    mode_of_travel : STRING
        'walking', 'sneaking', or 'crawling'.
    travel_time : FLOAT
        Uses speed_dic to calculate time between nodes.
    seeker_groups : DICT
        5 worst case seekers for every seeker passed in.

    Returns
    -------
    Maximum Visual Detection : FLOAT
        Returns the max visual detection for the worst case seekers
    
    '''
    
    for seeker in seekers:
        [seeker_coord, z, orient, orient_uncert] = seekers[seeker]
        distance_position_i=np.linalg.norm(np.array(seeker_coord)-position_i[:2])
        distance_position_j=np.linalg.norm(np.array(seeker_coord)-position_j[:2])
        if distance_position_i <=z or distance_position_j <= z:
            return 1

    visual_detection=[]
    for worst_case_seekers in seeker_groups:

        for seeker in seeker_groups[worst_case_seekers]:
            los_i = get_los(seeker, position_i) #Line of Sight to start postion
            los_j = get_los(seeker, position_j) #Line of sight to stop position
            los = (los_i+los_j)/2 #average line of sight from start to stop
            
            if los==0:
                visual_detection.append(0)
            else:
                [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
                seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
                alpha = get_alpha(seeker_coord, position_i, position_j, speed_dic[mode_of_travel])
                beta = get_beta(seeker_coord, position_i, position_j, height_dic[mode_of_travel])
                trace_ratio = closed_form_ratio(alpha, beta)
                # detection_probability = 999*trace_ratio/(998*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = 101*trace_ratio/(100*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = detection_over_step(int(travel_time), detection_probability) #testing this function out
                visual_detection.append(los*detection_probability)
    return max(visual_detection)

def get_los(seeker, evader_loc):
    if evader_in_blindspot(seeker, evader_loc):
        # print("blindspot")
        return 0
    [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
    seeker_loc=np.array([seeker_x, seeker_y, seeker_elevation])    
    
    "Create vector function r(t) from seeker to evader (the seeker's line of sight)"
    r0=seeker_loc[:]
    v=evader_loc-seeker_loc
    distance=np.linalg.norm(v)
    t=np.linspace(0,1,int(distance/(step_size/2.5))) #searches along route every ~step_size/2.5 meters
    r=[r0+v*t[i] for i in range(len(t))]
    
    "Find the vegetation factor on visibility"
    vegetation_factor=1
    for position in r:
        [x,y,e]=position
        ground_elevation = (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation
        if ground_elevation>e:
            "Line of sight blocked by obstace: Evader is in deadspace"
            # print("deadspace")
            return 0
        r = vegetation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]
        vegetation = (3-(3*(r/255)))
        vegetation_factor *= (1-(1/30)*vegetation) #assumes linear probability of seeing through vegetation probability = 1-(2/30)*density at any given point.
        #instances of seeing through vegetatin are independent, thus the probability of seeing through a bunch of vegetation of their multiplication.
        if vegetation_factor<.01:
            # print("VEGETATED")
            return 0
    return vegetation_factor


def get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups, testing=False):
    """
    This function should get the probability of audio detection
    Tasks:
        o Do the necessary calculations (Jakes function)
    """
    if testing:
        return 0
    x1 , y1, e1 = position_i 
    x2 , y2, e2 = position_j
    
    #print(x1 , y1, e1)
    
    x = (x1 + x2)/ 2
    y = (y1 + y2)/ 2
    
    audio_detection = []
    vegetation = (vegetation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*3
    #print(vegetation)
    
    point_source = np.array([vegetation,1])@point_source_dic[mode_of_travel] - 50
    
    #print(point_source)

    for worst_case_seekers in seeker_groups:
        for seeker in seeker_groups[worst_case_seekers]:

            [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
            seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
            seeker_to_evader = position_i-seeker_coord
            
            distance1 = np.linalg.norm(seeker_to_evader)
            distance2 = np.linalg.norm(position_j-seeker_coord)
            a = (probability_audio(point_source, distance1) + probability_audio(point_source, distance2)) / 2
            # print(a)
            audio_detection.append(a)
    
    #print(audio_detection)
    return max(audio_detection)

# p is the decibal level of moving through vegetation level

# flip the y values of the entire array and imagery

def probability_audio_old(point_source, dist):
    """
    Parameters
    ----------
    point_source : TYPE
        this is the decibal level of the evaders movement, dependent on the type of vegetation
    dist : TYPE
        DESCRIPTION.

    Returns
    -------
    final : TYPE
        DESCRIPTION.

    """
    r = dist
    exponent_1 = -(point_source-4)/20
    
    exponent_2 = 10*np.exp(exponent_1)*(r-np.exp(-exponent_1))
    
    final = 1/(1 + np.exp(exponent_2))
    
    #print(final)
    return final

def probability_audio(point_source, dist):
    dist = dist
    deltaDB = point_source - 2*np.log(dist)
    k = .1
    
    return 1 / (1 + np.exp(-k*deltaDB))


def get_seeker_group(templated_seeker):
    """
    takes a templated seeker location and creates a list of worst case seeker information.
    Assumes seeker is 2m tall

    Parameters
    ----------
    templated_seeker : list
        [seeker_loc, loc_uncertainty, theta_orientation, theta_uncertainty].
    node_field_info : list
        [nodes_wide, nodes_long, step_size, file_name, max_elevation, node_field].

    Returns
    -------
    seeker_group : list of lists
        [ worst case seeker 1 info (as a list of x coordinate, y coordinate, elevation, left limit to vision, right limit to vision),
         worst case seeker 1 info, 
         ...,
         worst case seeker max elevation info].

    """
    [seeker_loc, loc_uncertainty, theta_orientation, theta_uncertainty] = templated_seeker
    seeker_loc=np.array(seeker_loc)
    seeker_box = [seeker_loc+loc_uncertainty*rotate(theta_orientation)@np.array([(-1)**i,(-1)**j]) for i in range(2) for j in range(2)] #Get locations of corner seekers
    orient_left=theta_orientation+theta_uncertainty
    orient_right=theta_orientation-theta_uncertainty
    "Create locations of corner seekers listed as (x,y,elevation)"
    seeker_group=[]
    for i in range(4):
        [x,y] = seeker_box[i]
        seeker_group.append([x, y, (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation+2, orient_left, orient_right])
        
    [x_center,y_center]=seeker_loc
    locations=[[(x_center-loc_uncertainty+i,y_center-loc_uncertainty+j) for i in range(2*loc_uncertainty)] for j in range(2*loc_uncertainty)]

    "Find location of highest seeker in seeker box and add to seeker box"
    e_max=-999
    for i in range(2*loc_uncertainty):
        for j in range(2*loc_uncertainty):
            [x2,y2] = locations[i][j]
            e=(elevation_map.getpixel((x2*map_width_scale,y2*map_length_scale))[0]/255)*max_elevation
            if e>e_max:
                e_max=e
                [loc_x,loc_y]=locations[i][j]
    max_elevation_seeker = [loc_x, loc_y, e_max+2, orient_left, orient_right]
    seeker_group.append(max_elevation_seeker)
    
    return seeker_group

def rotate(theta):
    """
    2D rotational matrix to rotate 2D coordinates theta radians

    Parameters
    ----------
    theta : float
        desired rotation in radians.

    Returns
    -------
    R_theta : numpy array
        rotational matrix.

    """
    R_theta=np.array([[np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]])
    return R_theta


def evader_in_blindspot(seeker, evader_loc):
    [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
    ll = pos_angle(orient_left)
    rl = pos_angle(orient_right)
    seeker=np.array([seeker_x,seeker_y])
    evader=evader_loc[:2]
    s_e=evader-seeker
    
    angle_to_evader = np.arctan2(s_e[1],s_e[0])
    if true_range_angle(angle_to_evader, ll, rl):
        return True
    
    return False

def pos_angle(angle):
    """
    Takes any angle and returns that angle mapped to [0,2pi]
    """
    if angle < 0:
        return pos_angle(angle+2*np.pi)
    elif angle > 2*np.pi:
        return pos_angle(angle-2*np.pi)
    return angle

def true_range_angle(alpha, angle1, angle2):
    """
    Calculates if an angle is between two angles. Returns Boolean.
    """
    alpha = pos_angle(alpha)
    angle2 = pos_angle(angle2-angle1)
    alpha = pos_angle(alpha-angle1)

    if alpha < angle2:
        return True
    return False

def get_alpha(seeker_node, start_node, end_node, speed):
    """
    Get the birds eye angle between seeker to start and seeker to end.

    Parameters
    ----------
    seeker_node : numpy array
        coordinates of the seeker (x, y, elevation).
    start_node : numpy array
        starting coordinates of the evader (x, y, elevation).
    end_node : numpy array
        ending coordinates of the evader (x, y, elevation).
    speed : float
        evader's maximum possible (read worst case) speed in m/s.

    Returns
    -------
    alpha : float
        birds eye angle between seeker to start and seeker to end (in radians).

    """
    n_1 = np.array(start_node[:2])
    n_2 = np.array(end_node[:2])
    s = np.array(seeker_node[:2])
    if np.array_equiv(n_1, n_2):
        "In this case the person is transitioning modes and alpha is the worst case width of the evader"
        s_perpendicular=rotate(np.pi/2)@s
        s_perpendicular_unit = s_perpendicular / np.linalg.norm(s_perpendicular)
        n_2=n_1+s_perpendicular_unit*.5
   
    v = n_2 - n_1
    a = n_1 - s
    b = a + speed * v/np.linalg.norm(v)
    a_unit= a / np.linalg.norm(a)
    b_unit= b / np.linalg.norm(b)
    alpha = np.arccos(a_unit @ b_unit)
    return alpha


def get_beta(seeker_node, start_node, end_node, height):
    """
    Gets vertical angle use to calculate trace.

    Parameters
    ----------
    seeker_node : numpy array
        coordinates of the seeker (x, y, elevation).
    start_node : numpy array
        starting coordinates of the evader (x, y, elevation).
    end_node : numpy array
        ending coordinates of the evader (x, y, elevation).
    height : float
        height of the evader (tied to movement mode).

    Returns
    -------
    beta : float
        worst case vertical angle trace of evader during movement.

    """
    maximum_height=max(start_node[-1],end_node[-1])
    minimum_height=min(start_node[-1]-height,end_node[-1]-height)
    node_sorter=[start_node, end_node]
    distances=[np.linalg.norm(seeker_node-start_node), np.linalg.norm(seeker_node-end_node)]
    [x_start, y_start, elevation_start] = node_sorter[np.argmin(distances)]
    n1 = np.array([x_start, y_start, minimum_height])
    n2 = np.array([x_start, y_start, maximum_height])
    a = n1 - seeker_node
    b = n2 - seeker_node
    a_unit= a / np.sqrt(a @ a)
    b_unit= b / np.sqrt(b @ b)
    beta = np.arccos(a_unit @ b_unit)
    return beta

def closed_form_ratio(alpha, beta):
    evader_trace_1 = 2 * np.pi
    evader_trace_2 = np.arcsin(
        ((np.cos(alpha / 2)) * np.tan(beta / 2)) / np.sqrt((np.tan(alpha / 2) ** 2) + (np.tan(beta / 2) ** 2)))
    evader_trace_3 = np.arcsin(
        ((np.cos(beta / 2)) * np.tan(alpha / 2)) / np.sqrt((np.tan(alpha / 2) ** 2) + (np.tan(beta / 2) ** 2)))
    evader_trace = evader_trace_1 - 4 * (evader_trace_2 + evader_trace_3)
    seeker_visual_field = 3.7505
    closed_form = evader_trace / seeker_visual_field
    return min(1, closed_form)

def detection_over_step(steps, probability_single_step):
    # start_time=time.time()
    total_probability=sum([((-1)**i)*math.comb(steps,i+1)*(probability_single_step**(i+1)) for i in range(steps)])
    # print(time.time()-start_time)
    return total_probability  

# ----------------------------

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


def get_master_id(node_id, w, h, scale):
    """
    Given node in node id format, return master node number

    Parameters
    ----------
    node_id : STR. Requested node in the format 'm#'
    w : INT. Width of the node field 
    h : INT. height of the node field
    scale : FLOAT. distance between nodes
    """
    node_dic = {'w': 2*w*h, 's': w*h, 'c': 0}
    return node_dic[node_id[0]]+int(node_id[1:])

def adjacency_matrix(w,h,type_map,scale):
    """
    Generates adjacency matrix for w*h node field as a numpy array

    Parameters
    ----------
    w : INT. Width of the node field 
    h : INT. height of the node field
    type_map : STR. Map name.
    scale : FLOAT. distance between nodes

    Returns
    -------
    TYPE
        DESCRIPTION.
    """

    wh=w*h
    pos=['c','s','w']
    wh4=w*h*3
    pos_dic={'c':0,'s':wh,'w':2*wh}
    
    
    'create a w*h*3 by w*h*3 zero matrix'
    E=zerom(wh4,wh4)
    
    'Generate all possible edges'
    for p in pos:
        for nodei in range(wh):
            adjacents=adjacent_nodes(p+str(nodei+1),w,h,scale)
            node_id=get_node_id_master(nodei+1,w,h)
            niv=get_node_vector(node_id, w, h, scale)
            # nodei_elev=elevation_data(niv.x, niv.y, type_map,scale)
            
            for adj in adjacents:
                adjv=get_node_vector(adj, w, h,scale)
                theta=niv.angle_to(adjv)
                #steps=np.linspace(0,adjv.distance(niv),scale)
                steps = np.linspace(0, adjv.distance(niv), int(scale))

                # adj_elevs=[nodei_elev]
                path_possible=True
                # for step in steps:
                #     adj_elevs.append(elevation_data(niv.x+step*np.cos(theta), niv.y+step*np.sin(theta), type_map,scale))
                #     if abs(adj_elevs[-1]-adj_elevs[-2])>=10:
                #         path_possible=False
                #         print(node_id,niv,adjv,abs(adj_elevs[-1]-adj_elevs[-2]),steps,adjacents)
                #         break
                if path_possible:
                    E[nodei+pos_dic[p]][pos_dic[adj[0]]+int(adj[1:])-1]=1
    return np.array(E)

def zerom(m,n):
    """
    Generates an m by n matrix with all elements 0
    """
    zeroM=[]
    for i in range(m):
        zeroM.append([])
        for j in range(n):
            zeroM[i].append(0)
    return zeroM

class Vector_2D(object):
    def __init__(self,x,y):
        
        self.x=x
        self.y=y
        self.v=[self.x,self.y]
    
    def mag(self):
        """
        Caluculates the magnitude of the vector
        """
        return np.sqrt(self.x**2+self.y**2)
    
    def __add__(self,other):
        """
        For Adding two vectors
        """
        return Vector_2D(self.x+other.x,self.y+other.y)
    
    def __sub__(self,other):
        """
        For Subtracting two vectors
        """
        return self+(other*-1)
    
    def __mul__(self,other):
        """
        For Multiplying two vectors or a vector and a scalar
        """
        if isinstance(other,(int,float)):
            return Vector_2D(self.x*other,self.y*other)
        else:
            return self.x*other.x+self.y*other.y
    
    def angle_between(self,other):
        """
        Calculate the angle between two vectors
        """
        self_mag=self.mag()
        other_mag=other.mag()
        if self_mag==0:
            return np.arctan2(other.y,other.x)
        elif other_mag==0:
            return np.arctan2(-self.y,-self.x)
        else:
            dot_mag=self*other/(self_mag*other_mag)
            return np.arccos(dot_mag)
        
    def angle_to(self,other):
        """
        Calculate the angle from vector 1 to vector 2
        """
        diff=other-self
        return np.arctan2(diff.y,diff.x)
        
    def distance(self,other):
        """
        Calculate the distance from vector 1 to vector 2
        """
        return (other-self).mag()
    
    def __truediv__(self,other):
        """
        Allow scalar division
        """
        if isinstance(other,(int,float)):
            return self*(1/other)
        else:
            print("Can only divide a vector by a scalar")
            return
        
    def __repr__(self):
        """
        For pretty printing
        """
        printed="<"+str(self.x)+" , "+str(self.y)+">"
        return printed
    
    def __eq__(self,other):
        """
        Return boolean value for whether vector 1 is vector 2
        """
        return self.x==other.x and self.y==other.y
    


def get_node_id_from_master(master_id, w, h):
    wh = w * h
    if master_id <= wh:
        return 'c' + str(master_id)
    elif master_id <= 2 * wh:
        return 's' + str(master_id - wh)
    elif master_id <= 3 * wh:
        return 'w' + str(master_id - 2 * wh)
    else:
        return 'Invalid master_id'
    
def get_movement_method(current, next_node):
    # This function determines the movement method based on the current and next nodes.
    # For simplicity, let's assume the movement method is encoded in the node ID (e.g., 'w' for walk, 'c' for crouch, etc.)
    # You can modify this function based on your actual implementation.

    node_id = get_node_id_from_master(next_node, w, h)
    if node_id.startswith('w'):
        return 'w'
    elif node_id.startswith('c'):
        return 'c'
    elif node_id.startswith('s'):
        return 's'
    else:
        return 'walk'  # Default to walk if unknown
    


def adjacency_matrix(w,h,type_map,scale):
    """
    Generates adjacency matrix for w*h node field as a numpy array

    Parameters
    ----------
    w : INT. Width of the node field 
    h : INT. height of the node field
    type_map : STR. Map name.
    scale : FLOAT. distance between nodes

    Returns
    -------
    TYPE
        DESCRIPTION.
    """

    wh=w*h
    pos=['c','s','w']
    wh4=w*h*3
    pos_dic={'c':0,'s':wh,'w':2*wh}
    
    
    'create a w*h*3 by w*h*3 zero matrix'
    E=zerom(wh4,wh4)
    
    'Generate all possible edges'
    for p in pos:
        for nodei in range(wh):
            adjacents=adjacent_nodes(p+str(nodei+1),w,h,scale)
            node_id=get_node_id_master(nodei+1,w,h)
            niv=get_node_vector(node_id, w, h, scale)
            # nodei_elev=elevation_data(niv.x, niv.y, type_map,scale)
            
            for adj in adjacents:
                adjv=get_node_vector(adj, w, h,scale)
                theta=niv.angle_to(adjv)
                #steps=np.linspace(0,adjv.distance(niv),scale)
                steps = np.linspace(0, adjv.distance(niv), int(scale))

                # adj_elevs=[nodei_elev]
                path_possible=True
                # for step in steps:
                #     adj_elevs.append(elevation_data(niv.x+step*np.cos(theta), niv.y+step*np.sin(theta), type_map,scale))
                #     if abs(adj_elevs[-1]-adj_elevs[-2])>=10:
                #         path_possible=False
                #         print(node_id,niv,adjv,abs(adj_elevs[-1]-adj_elevs[-2]),steps,adjacents)
                #         break
                if path_possible:
                    E[nodei+pos_dic[p]][pos_dic[adj[0]]+int(adj[1:])-1]=1
    return np.array(E)

def get_D(w,h,step):
    E=adjacency_matrix(w,h,0,step)
    n=3*w*h
    num_enemy=random.randint(1,5)
    D=np.zeros((n,n))
    enemies={}
    for i in range(num_enemy):
        enemies[i+1]=[np.array([w*step*random.random(),h*step*random.random()]),5+40*random.random()]
    for i in range(3*w*h):
        sn= get_node_id_master(i+1,w,h)
        start=np.array(get_node_vector(sn,w,h,step).v)
        rest = [i+k for k in range(3*w*h-i)]
        for j in rest:
            fn = get_node_id_master(j+1,w,h) 
            stop = np.array(get_node_vector(fn,w,h,step).v)
            
            if E[i][j]==1:
                detection=false_detection(step, start, stop, enemies, fn)
                D[i][j]=detection
                D[j][i]=detection
#                 print(start, stop,detection)
        # print(E[i])
        # print(D[i])
    return D, enemies

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def index_to_coordinates(index, width):
    # Convert a single index to x, y coordinates based on the width of the grid
    return (index % width, index // width)


def a_star_search_hueristic(distance_matrix, start, goal,time_limit=3600):
    avg_detection = np.mean(distance_matrix)

    frontier = [(0, start, 0)]  # Added time elapsed as third element
    came_from = {start: None}
    cost_so_far = {start: 0}
    time_so_far = {start: 0}  # Track time elapsed for each node

    while frontier:
        _, current, time_elapsed = heapq.heappop(frontier)

        if current == goal or time_elapsed > time_limit:
            break

        for i, weight in enumerate(distance_matrix[current]):
            if weight == 0:
                continue

            # Calculate time cost based on movement method
            movement_method = get_movement_method(current, i)  # You'll need to implement this function
            time_cost = 10/MOVEMENT_SPEEDS[movement_method]

            new_time_elapsed = time_elapsed + time_cost
            if new_time_elapsed > time_limit:
                continue  # Skip this action if it exceeds the time limit

            alpha = 0.5
            detection_penalty = alpha * np.max(D_for_Astar)

            new_cost = cost_so_far[current] + weight + (detection_penalty * D_for_Astar[current][i])

            if i not in cost_so_far or new_cost < cost_so_far[i]:
                cost_so_far[i] = new_cost
                time_so_far[i] = new_time_elapsed
                current_coords = index_to_coordinates(current, w)
                goal_coords = index_to_coordinates(goal, w)
                heuristic_value = euclidean_distance(current_coords, goal_coords) + (avg_detection * distance_matrix[current][i])
                priority = new_cost + heuristic_value
                heapq.heappush(frontier, (priority, i, new_time_elapsed))
                came_from[i] = current


    # Reconstruct path
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path, cost_so_far[goal]

def smooth_path(path):
    i = 0
    while i < len(path) - 2:
        pointA = np.array(index_to_coordinates(path[i], w))
        pointB = np.array(index_to_coordinates(path[i+1], w))
        pointC = np.array(index_to_coordinates(path[i+2], w))
        
        direct_distance = np.linalg.norm(pointA - pointC)
        detour_distance = np.linalg.norm(pointA - pointB) + np.linalg.norm(pointB - pointC)
        
        if abs(direct_distance - detour_distance) < 1e-2:  # Threshold can be adjusted
            path.pop(i+1)
        else:
            i += 1
    return path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# D_for_Astar, enemies=get_D(10,10,10)
# start_time = time.time()
# path, cost = a_star_search_hueristic(D_for_Astar, start_node, goal_node)
# path = smooth_path(path)
# end_time = time.time()
# a_star_time = end_time - start_time

# plot_path(path, w, h, scale)
# print("A-star")
# print(f"Path: {path}")
# print(f"Cost: {cost}")
# print(f"Time taken by A-star with hueristic: {a_star_time:.4f} seconds")

import numpy as np

def arc_ID(w, h):
    arc = []
    left_edge = np.linspace(w+1, (h-2)*w+1, h-2)
    right_edge = np.linspace(2*w, (h-1)*w, h-2)
    bottom_edge = np.linspace(2, w-1, w-2)
    top_edge = np.linspace((h-1)*w+2, w*h-1, w-2)
    bottom_left = [1, w*h+1, 2*w*h+1]
    bottom_right = [w, w*h+w, 2*w*h+w]
    top_left = [w*(h-1)+1, w*h+w*(h-1)+1, 2*w*h+w*(h-1)+1]
    top_right = [w*h, 2*w*h, 3*w*h]
    for i in range(2):
        left_edge = np.append(left_edge, np.linspace(
            (i+1)*w*h+w+1, (i+1)*w*h+(h-2)*w+1, h-2))
        right_edge = np.append(right_edge, np.linspace(
            (i+1)*w*h+2*w, (i+1)*w*h+(h-1)*w, h-2))
        bottom_edge = np.append(bottom_edge, np.linspace(
            (i+1)*w*h+2, (i+1)*w*h+w-1, w-2))
        top_edge = np.append(top_edge, np.linspace(
            (i+1)*w*h+(h-1)*w+2, (i+1)*w*h+w*h-1, w-2))

    lvl = w*h
    A = 3*w*h
    a = 1
    arc_dic = {}
    for i in range(A):
        k = i+1
        if k > lvl:
            arc_dic[a] = (k, k-lvl)
            a += 1

        if k in bottom_left:
            adj = [k+1, k+w, k+1+w]

        elif k in bottom_right:
            adj = [k-1, k-1+w, k+w]

        elif k in bottom_edge:
            adj = [k-1, k+1, k+w-1, k+w, k+w+1]

        elif k in left_edge:
            adj = [k-w, k-w+1, k+1, k+w, k+w+1]

        elif k in right_edge:
            adj = [k-w-1, k-w, k-1, k+w-1, k+w]

        elif k in top_edge:
            adj = [k-w-1, k-w, k-w+1, k-1, k+1]

        elif k in top_left:
            adj = [k-w, k-w+1, k+1]

        elif k in top_right:
            adj = [k-w-1, k-w, k-1]

        else:
            adj = [k-w-1, k-w, k-w+1, k-1, k+1, k+w-1, k+w, k+w+1]

        for n in adj:
            arc_dic[a] = (k, n)
            a += 1

        if k <= 2*lvl:
            arc_dic[a] = (k, k+lvl)
            a += 1
        print(str(i+1)+" of "+str(A)+" complete")
    return arc_dic



"""
-------------------------------------------------------------------------------
Visualization Functions
-------------------------------------------------------------------------------
"""

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


def plot_path(situtation_name, start, end, seekers=seekers, w=nodes_wide,h=nodes_long, scale=10, detection=False):
    single_field = w*h 

    plot_contour()
    if detection == True:
        detection_fields('charging')

    path = read_XK(start, end, "output_"+situtation_name +".csv", w, h)
    print('Paths Followed:',path)

    
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

    # plt.plot(pathx, pathy, color='black', linewidth=3)

    # Add arrows to indicate direction
    for i in range(len(pathx) - 1):
    # Check if the segment is in the special set
        segment = (path[i], path[i + 1])
        # print(path[i])
        if (path[i] > single_field) or (path[i]+1 > single_field):
            color = 'green'
        else:
            color = 'black'
        
        # Arrows
        #plt.annotate('', xy=(pathx[i + 1], pathy[i + 1]), xytext=(pathx[i], pathy[i]),
        #            arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.05))
        
        # Line Segments
        plt.plot([pathx[i], pathx[i + 1]], [pathy[i], pathy[i + 1]], color=color, linewidth=3)

        
    plt.plot([], [], color='green', label='charging')  # Placeholder for legend
    plt.plot([], [], color='black', label='charged')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Path with Direction')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_energy(situtation_name, seekers=seekers, w=nodes_wide,h=nodes_long, scale=10):
    energyStep = []
    elevationStep = []
    arcPath = []

    arc_dic = {}
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip header if present
        next(csv_reader, None)
        for row in csv_reader:
            key = int(row[0])  # Assuming the first column is the key
            energy = float(row[5])
            elevation = float(row[-1])   # Assuming the rest of the row are the values
            arc_dic[key] = [energy, elevation]

    with open("output_fixed.csv", mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            arcPath.append(float(row[0]))
            
    for arc in arcPath:
        energyStep.append(arc_dic[arc][0])
        elevationStep.append(arc_dic[arc][1])

    # Calculate the remaining battery after each step
    remaining_battery = [initial_battery]  # Start with full battery

    for spent in energyStep:
        remaining_battery.append(remaining_battery[-1] - spent)

    # Remove the initial value since we want to plot the usage after steps
    remaining_battery.pop(0)

    # Plot the remaining battery vs step
    plt.plot(range(len(remaining_battery)), remaining_battery, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Remaining Battery Charge')
    plt.title('Battery Expenditure Over Steps')

    # Display the plot
    plt.show()

    plt.plot(range(len(elevationStep)), elevationStep, marker='x', label='Elevation Step')

    # Bar plot for energy steps
    plt.bar(range(len(energyStep)), energyStep, color='blue', alpha=0.6, label='Energy Step')

    # Add labels and title
    plt.ylabel('Energy Cost')
    plt.xlabel('Step')
    plt.title('Energy Cost per Step')

    # Display the plot
    plt.legend()
    plt.show()

    print('Energy Per Step')
    print(energyStep)
    print('Total Energy Expended:',sum(energyStep))


    # print(elevationStep)
    # plt.ylabel('Elevation')
    # plt.xlabel('Step')
    # plt.title('Elevation per Step')


    charging = 0
    for i in arcPath:
        if i >= 45602/2:
            charging += 1
    print(charging)

    print(arcPath)


def calculateDetection(situation_name):
    detectionStep = []
    arcPath = []
    pathFile = 'output_'+situation_name+'.csv'

    arc_dic = {}
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip header if present
        next(csv_reader, None)
        for row in csv_reader:
            key = int(row[0])  # Assuming the first column is the key
            energy = float(row[5])
            elevation = float(row[-1])
            detection = float(row[3])   # Assuming the rest of the row are the values
            arc_dic[key] = [energy,detection, elevation]

    with open(pathFile, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            arcPath.append(float(row[0]))
            
    for arc in arcPath:
        detectionStep.append(arc_dic[arc][1])

    # Calculate 
    no_detection_prob = np.prod([1 - p for p in detectionStep])
    # Return the probability of detection
    return 1 - no_detection_prob
    