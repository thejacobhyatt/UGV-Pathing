# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:50:04 2023

@author: thomas.kendall
"""

import numpy as np
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
from tqdm import tqdm
import csv
import random as random
import pandas as pd
"""
-------------------------------------------------------------------------------
Universals
-------------------------------------------------------------------------------
"""
speed_dic = {'walking': 1.4, 'crawling': .1, 'sneaking': .75, 'crawling_sneaking':.5, 'walking_sneaking':1, 'robo': 1} #VALIDATE THESE SPEEDS THROUGH TESTING!
height_dic = {'walking': 1.8, 'crawling': 1, 'sneaking': .15, 'crawling_sneaking':1, 'walking_sneaking':1.8} #CONFIRM HEIGHTS
transition_time_dic = {'crawling_sneaking': 2, 'walking_sneaking': .5} #CONFIRM THESE TIMES IN SECONDS!
seeker_orientation_uncertainty = {'human': (62*np.pi/180)+np.pi/2, 'bunker': 62*np.pi/180} #Enemy capabilities will influence this (among other things), will likely need to expand dictionary and definitions for each key
"""
-------------------------------------------------------------------------------
User Input
-------------------------------------------------------------------------------
"""
#Relative PATH in GitHub Project
file_path = '..\\OTP Figures\\'
file_name=file_path+"Buckner" #map identifier
desired_lower_left_corner = (200, 200) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
desired_upper_right_corner = (500, 500) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
step_size=10  #desired distance, in meters, between nodes CANNOT BE TOO LARGE OR WILL CAUSE OverflowError when determining probability
# seekers ={1: [(5,5), 5, 0, seeker_orientation_uncertainty['human']]}
seekers={1 : [(50,50), 15, 0, seeker_orientation_uncertainty['human']], 2 : [(150,275), 15, -np.pi/2, seeker_orientation_uncertainty['human']], 3 : [(250,50), 10, 3*np.pi/4, seeker_orientation_uncertainty['bunker']]}
# #{seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ], next seeker : [...], ...}
fog_coef = 0
"""
-------------------------------------------------------------------------------
MAJ Kendall Test Input
-------------------------------------------------------------------------------
"""
# file_path = '..\\OTP Figures\\'
# file_name=file_path+"Buckner" #map identifier
# desired_lower_left_corner = (200, 200) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
# desired_upper_right_corner = (290, 290) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
# step_size=10  #desired distance, in meters, between nodes CANNOT BE TOO LARGE OR WILL CAUSE OverflowError when determining probability
# seekers={1 : [(100/2,100/2), 15, 0, seeker_orientation_uncertainty['bunker']]}#, 2 : [(150,275), 15, -np.pi/2, seeker_orientation_uncertainty['human']], 3 : [(50,50), 10, np.pi/4, seeker_orientation_uncertainty['bunker']]}
#{seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ], next seeker : [...], ...}
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

def my_function_to_test():
    return 1+1

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
    arcs, node_field = get_arcs2()
    N = len(node_field)
    ordered_arcs = get_ordered_arcs(arcs, node_field, N)
    
    'Write Arcs file to csv'
    crawling_arcs = []
    sneaking_arcs = []
    arc_name = 'arcs_' + middle_name + '_' + scenario_name + '.csv'
    with open(arc_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for arc in ordered_arcs:
            (node_i, node_j) = arc
            [arc_ID, risk, time, movement_code] = ordered_arcs[arc] 
            if movement_code == 1:
                crawling_arcs.append(arc_ID)
            elif movement_code == 2:
                sneaking_arcs.append(arc_ID)
            writer.writerow([arc_ID, node_i, node_j, risk, time, movement_code])
            
    "Write the crawling sneaking dataframe to a csv file"    
    group_name = middle_name+'_CS.csv'
    if path.exists(group_name):
        print("Crawling Sneaking Dataframe skipped: already written")
    else:
        crawling_sneaking_arcs_dataframe = np.transpose([crawling_arcs, sneaking_arcs])
        with open(group_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(crawling_sneaking_arcs_dataframe)

    "Write Node Field Structure csv files, if needed"
    tri_name = middle_name + '_triangle.csv'
    in_name = middle_name + '_inflow.csv'
    out_name = middle_name + '_outflow.csv'
    
    if path.exists(tri_name) and path.exists(out_name):
        print("Inflows, outflows, and triangle sets skipped: already written")
        return 
    
    "Get inflow sets, outflow sets, and triangle sets"
    node_i_inflows, node_i_outflows = get_ins_outs(ordered_arcs, N)
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
    travel_dic = {'walking' : 3, 'sneaking' : 2, 'crawling' : 1, 'crawling_sneaking' : 0, 'walking_sneaking' : 0} #Used to produce the movement code
    arc_ID = 1 #Initiate arc IDs
    ordered_arcs={} # Initiate ordered arcs dictionary
    for i in range(N): 
        node_i = i + 1
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        for node_j in adjacent_nodes_i:
            [ (x_i, y_i, z_i), (x_j, y_j, z_j), mode_of_travel, time, risk ] = arcs[(node_i, node_j)]
            # if node_i <= nodes_wide*nodes_long:
            #     print(arc_ID, node_i, node_j)
            ordered_arcs[(node_i, node_j)] = [arc_ID, risk, time, travel_dic[mode_of_travel]]
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
                    [arc_ij, risk, time, movement_code] = ordered_arcs[(node_i, node_j)] 
                    [arc_jk, risk, time, movement_code] = ordered_arcs[(node_j, node_k)] 
                    [arc_ik, risk, time, movement_code] = ordered_arcs[(node_i, node_k)] 
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
        [outflow_arc_ID, risk, time, movement_code] = ordered_arcs[outflow_arc] 
        [inflow_arc_ID, risk, time, movement_code] = ordered_arcs[inflow_arc] 
        node_i_inflows[node_i].append(inflow_arc_ID)
        node_i_outflows[node_i].append(outflow_arc_ID)
        
    return node_i_inflows, node_i_outflows
"""
-------------------------------------------------------------------------------
Solution Functions
-------------------------------------------------------------------------------
"""

def read_output(start, end, file_name, w, h):
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
    arc_dic = arc_ID(w, h)

    for arc in unsorted_arcs:
        (start_node, end_node) = arc_dic[arc]
        unsorted_path.append([start_node, end_node])

    print(unsorted_path)
    
    return 
    visited = []
    PATH = [start]
    
    while PATH[-1] != end:
        for node_pair in unsorted_path:
            print(node_pair)

            start_node = node_pair[1]

            if start_node == PATH[-1] and list(node_pair) not in visited:
                print(node_pair)
                PATH.append(int(node_pair[0]))
                visited.append(list(node_pair))
                break
            
    return PATH

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
        # print(str(i+1)+" of "+str(A)+" complete")
    return arc_dic

"""
-------------------------------------------------------------------------------
Visualization Functions
-------------------------------------------------------------------------------
"""

def detection_fields(mode_of_travel, perpendicular, plot_node_field, seekers=seekers):
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
            # visual_detection = get_visual_detection_2(position_i, mode_of_travel, travel_time, seeker_groups, perpendicular)
            visual_detection = 0 
            audio_detection = get_audio_detection(position_i, position_i, mode_of_travel, seeker_groups) #WILL NEED TO FIX
            #aud_lis.append(audio_detection)
            # audio_detection = 0
            # checked_locations+=1
            # print(progress_bar(np.round(checked_locations/total_locations, 2)))
            detection[-1].append(max(visual_detection,audio_detection))
    plt.clf()
    plt.style.use('ggplot')
    fig, ax= plt.subplots()
    im = ax.imshow(detection, extent=[0, xvals[-1], 0, yvals[-1]],
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
        node_field=create_node_field()
        for i in range(nodes_wide*nodes_long):
            (x,y)=node_field[i+1][0]
            plt.scatter(x,y,color="black")
    # plt.title('Detection Radius for Multiple Enemies')
    return     


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
    plt.clf()
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

def visualize_node_field(requested_mode, plot_nodes):
    
    colors_dic={'walking': 'red', 'crawling': 'blue', 'sneaking': 'green', 'crawling_sneaking':'black', 'walking_sneaking':'black'}
    
    arcs=get_arcs()
    plotted_arcs=[]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for arc in arcs:
        (node_i, node_j) = arc
        reverse_arc = (node_j, node_i)
        if reverse_arc not in plotted_arcs:
            [ (x_i, y_i, z_i), (x_j, y_j, z_j), mode_of_travel, time, risk ] = arcs[arc]
            # print(z_i, z_j)
            if mode_of_travel == requested_mode or requested_mode=='all':
                if plot_nodes:
                    ax.scatter(x_i, y_i, z_i, color='black')
                    ax.scatter(x_j, y_j, z_j, color='black')
                plt.plot([x_i,x_j],[y_i,y_j],[z_i,z_j],color=colors_dic[mode_of_travel])
                plotted_arcs.append(arc)
                plotted_arcs.append(reverse_arc)
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                Line2D([0], [0], color='blue', lw=1),
                Line2D([0], [0], color='green', lw=1),
                Line2D([0], [0], color='black', lw=1)]
    plt.legend(custom_lines, ['walking', 'crawling', 'sneaking', 'chaning'])
    return
        
"""
-------------------------------------------------------------------------------
Main Functions
-------------------------------------------------------------------------------
"""

def write_csvs_for_jason(arc_dic,file_name):
    adj_name=file_name+'_adj.csv'
    det_name=file_name+'_det.csv'
    time_name=file_name+'_time.csv'
    N=3*nodes_wide*nodes_long
    adj_M=np.zeros((N,N))
    det_M=np.zeros((N,N))
    time_M=np.zeros((N,N))
    for arc in arc_dic:
        # print(arc)
        (i,j)=arc
        [position_i, position_j, mode_of_travel, travel_time, risk_level] = arc_dic[arc]
        adj_M[i-1,j-1]=1
        det_M[i-1,j-1]=risk_level
        time_M[i-1,j-1]=travel_time
    with open(adj_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adj_M)
    print("hi")
    with open(det_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(det_M)
    print("hello there")
    with open(time_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(time_M)
    return 
    
def get_arcs(nodes_wide=nodes_wide, nodes_long=nodes_long, step_size=step_size, file_name=file_name, max_elevation=max_elevation, seekers=seekers):
    """
    gets arcs

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
    """
    # start_time = time.time()
    
    node_field=create_node_field()
    
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker]) for templated_seeker in seekers}
    
    single_field=nodes_wide*nodes_long
    crawling_nodes=[i+1 for i in range(single_field)]
    sneaking_nodes=[i+1+single_field for i in range(single_field)]
    walking_nodes=[i+1+2*single_field for i in range(single_field)]
    
    "Create arcs dictionary"
    arcs={}
    # arc_length=get_arc_length()/2
    N=3*single_field

    checked=0
    for i in tqdm(range(N)):
        node_i=i+1
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        position_i=np.array(coordinate_i+(elevation_i,))
        node_i_land=classify_node(position_i)
        for node_j in adjacent_nodes_i:
            "Dont recaclulate arcs arleady found"
            key=(node_j,node_i)
            if key not in arcs:
                [coordinate_j, elevation_j, vegetation_j, adjacent_nodes_j] = node_field[node_j]
                position_j=np.array(coordinate_j+(elevation_j,))
                node_j_land=classify_node(position_j)
                
                "Only add Arc if arc is possible (no cliff and neither node is in water)"
                if abs(elevation_i-elevation_j)<=2*step_size and node_i_land and node_j_land: #WHAT IS THE STEEPEST GRADE A HUMAN CAN CLIMB
                    
                    "Determine Time for travel"
                    if coordinate_i == coordinate_j:
                        if (node_i in crawling_nodes and node_j in sneaking_nodes) or (node_j in crawling_nodes and node_i in sneaking_nodes):
                            mode_of_travel='crawling_sneaking'
                            
                        else:
                            mode_of_travel='walking_sneaking'
                        travel_time=transition_time_dic[mode_of_travel]
                    else:
                        distance = np.linalg.norm(position_j-position_i)
                        if node_i in walking_nodes:
                            mode_of_travel='walking'
                        elif node_i in sneaking_nodes:
                            mode_of_travel='sneaking'
                        else:
                            mode_of_travel='crawling'
                        
                        "Determine time base on speed, and average vegetation encountered"
                        average_vegetation=(vegetation_i+vegetation_j)/2
                        vegetation_factor=1-(2/9)*average_vegetation #NEEDS VALIDATION THROUGH TESTING!!!!!!!!!!!
                        travel_time=distance/(vegetation_factor*speed_dic[mode_of_travel])
                    
                    "Determine Probability of detection"
                    visual_detection=get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups)
                    audio_detection=get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups)
                risk_level = max(visual_detection,audio_detection)    
                arcs[(node_i,node_j)]=[position_i, position_j, mode_of_travel, travel_time, risk_level]
                arcs[(node_j,node_i)]=[position_j, position_i, mode_of_travel, travel_time, risk_level]
                checked+=1
                # print(node_i,"-> ", node_j,"complete, risk=", np.round(risk_level,4), "Status:",np.round(100*(checked)/arc_length,2),"% Complete")
    # print("Calculation Time:",time.time()-start_time)
    return arcs

def get_arcs2(nodes_wide=nodes_wide, nodes_long=nodes_long, step_size=step_size, file_name=file_name, max_elevation=max_elevation, seekers=seekers):
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
    
    node_field=create_node_field()
    
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker]) for templated_seeker in seekers}
    
    single_field=nodes_wide*nodes_long
    crawling_nodes=[i+1 for i in range(single_field)]
    sneaking_nodes=[i+1+single_field for i in range(single_field)]
    walking_nodes=[i+1+2*single_field for i in range(single_field)]
    
    "Create arcs dictionary"
    arcs={}
    # arc_length=get_arc_length()/2
    N=3*single_field

    checked=0
    for i in tqdm(range(N)):
        node_i=i+1
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        position_i=np.array(coordinate_i+(elevation_i,))
        node_i_land=classify_node(position_i)
        for node_j in adjacent_nodes_i:
            "Dont recaclulate arcs arleady found"
            key=(node_j,node_i)
            if key not in arcs:
                [coordinate_j, elevation_j, vegetation_j, adjacent_nodes_j] = node_field[node_j]
                position_j=np.array(coordinate_j+(elevation_j,))
                node_j_land=classify_node(position_j)
                
                "Only add Arc if arc is possible (no cliff and neither node is in water)"
                if abs(elevation_i-elevation_j)<=2*step_size and node_i_land and node_j_land: #WHAT IS THE STEEPEST GRADE A HUMAN CAN CLIMB
                    
                    "Determine Time for travel"
                    if coordinate_i == coordinate_j:
                        if (node_i in crawling_nodes and node_j in sneaking_nodes) or (node_j in crawling_nodes and node_i in sneaking_nodes):
                            mode_of_travel='crawling_sneaking'
                            
                        else:
                            mode_of_travel='walking_sneaking'
                        travel_time=transition_time_dic[mode_of_travel]
                    else:
                        distance = np.linalg.norm(position_j-position_i)
                        if node_i in walking_nodes:
                            mode_of_travel='walking'
                        elif node_i in sneaking_nodes:
                            mode_of_travel='sneaking'
                        else:
                            mode_of_travel='crawling'
                        
                        "Determine time base on speed, i average vegetation encountered"
                        average_vegetation=(vegetation_i+vegetation_j)/2
                        vegetation_factor=1-(2/9)*average_vegetation #NEEDS VALIDATION THROUGH TESTING!!!!!!!!!!!
                        travel_time=distance/(vegetation_factor*speed_dic[mode_of_travel])
                    
                    "Determine Probability of detection"
                    visual_detection=get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups)
                    audio_detection=get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups)
                risk_level = max(visual_detection,audio_detection)    
                arcs[(node_i,node_j)]=[position_i, position_j, mode_of_travel, travel_time, risk_level]
                arcs[(node_j,node_i)]=[position_j, position_i, mode_of_travel, travel_time, risk_level]
                checked+=1
                # print(node_i,"-> ", node_j,"complete, risk=", np.round(risk_level,4), "Status:",np.round(100*(checked)/arc_length,2),"% Complete")
    # print("Calculation Time:",time.time()-start_time)
    return arcs, node_field

def create_node_field():
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
            #crawling level
            node_field[node_id] = [coordinate, elevation+.15, vegetation, get_adjacent_nodes(node_id, coordinate)]
            
            #sneaking level
            node_field[node_id+single_field] = [coordinate, elevation+1, vegetation, get_adjacent_nodes(node_id+single_field, coordinate)]
            
            #walking level
            node_field[node_id+2*single_field] = [coordinate, elevation+1.8, vegetation, get_adjacent_nodes(node_id+2*single_field, coordinate)]
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
        in_horizon = potential_adjacents[i]>0 and potential_adjacents[i]<=3*single_field
        in_map = potential_location[0]>=0 and potential_location[0]<step_size*nodes_wide and potential_location[1]>=0 and potential_location[1]<step_size*nodes_long
        if in_horizon and in_map:
            actual_adjacents.append(potential_adjacents[i])

            
    return actual_adjacents

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
    
point_source_dic ={'walking' : np.array([40/3, 20]), 'sneaking' : np.array([30/3, 15]), 'crawling' : np.array([20/3, 10]), 'crawling_sneaking':np.array([30/3, 15])}


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
    

"""
-------------------------------------------------------------------------------
Weather Modifiers
-------------------------------------------------------------------------------
"""    

def wind_multiplier(seeker_coord, position_i, wind_direction):
    multiplier = 2
    numerator = (s - e).norm() * wind_direction.norm() + 1
    
    wind_scaler = multiplier* abs(wind_direction) * (numerator/2)
    
    return wind_scaler

def audio_multiplier():
    if chance_snow > 0:
        min_bound = 1
        max_bound = .5
        return min_bound + (max_bound - min_bound) * (chance_snow - min(chance_snow)) / (max(chance_snow) - min(chance_snow))
   
    elif chance_rain > 0: 
        min_bound = 1
        max_bound = .25
        return min_bound + (max_bound - min_bound) * (chance_rain - min(chance_rain)) / (max(chance_rain) - min(chance_rain))
    
    

    
"""
-------------------------------------------------------------------------------
Supporting Functions
-------------------------------------------------------------------------------
"""    
    
def get_arc_length():
    
    return 12 - 18 * (nodes_wide+nodes_long)+28 * nodes_wide * nodes_long

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

def classify_node(node_coordinate):
    "This function should check if node is land (True) or water (False)"
    x, y, elevation = node_coordinate
    
    if vegetation_map.getpixel((x*map_width_scale,y*map_length_scale))[0] > 235:
        return False
    
    return True

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

def pad_to_length(data, length):
    if len(data) < length:
        data.extend([0] * (length - len(data)))
    return data


"""
-------------------------------------------------------------------------------
Plot the Solution
-------------------------------------------------------------------------------
"""

def heuristic(node1, node2,desired_map_length,desired_map_width):
    x1, y1 = node1[0]
    x2, y2 = node2[0]
    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    max_distance = ((desired_map_width)**2 + (desired_map_length)**2)**0.5  # Adjust as necessary
    return distance / max_distance



"""
Ideas for improvement:
-find the quickest path as well as the least detection path-- specifically, solve for path that minimizes probability of detection, then substitute probability of detection as the main constraint and find the  path that minimizes travel time given that constraint.
-add constraints to limit the amount of time spent crawling and sneaking
-change colors based on walking crawling or sneaking
-report the highest probability node
"""

def a_star(start, goal, arcs, node_field, time_constraint,desired_map_length,desired_map_width,hueristic_on=False):
    open_set = []
    heapq.heappush(open_set, (0, start, 0, 0))  # (cost, current_node, travel_time, probability_of_detection)
    came_from = {}
    cost_so_far = {start: 0}
    time_so_far = {start: 0}
    detection_so_far = {start: 0}

    while open_set:
        _, current, current_time, current_detection = heapq.heappop(open_set)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, cost_so_far[goal], time_so_far[goal], detection_so_far[goal]

        for next_node in node_field[current][3]:  # Adjacent nodes are in the 4th position
            travel_time = current_time + arcs[(current, next_node)][3]  # Travel time is in the 4th position
            detection_probability = current_detection + arcs[(current, next_node)][4]  # Detection probability is in the 5th position

            if travel_time <= time_constraint:
                new_cost = detection_so_far[current] + arcs[(current, next_node)][4]  # Use detection probability as cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    time_so_far[next_node] = travel_time
                    detection_so_far[next_node] = detection_probability
                    if hueristic_on==False:
                        
                        priority = new_cost #+ heuristic(node_field[next_node], node_field[goal],desired_map_length,desired_map_width)
                    else:
                        priority = new_cost + heuristic(node_field[next_node], node_field[goal],desired_map_length,desired_map_width)

                    heapq.heappush(open_set, (priority, next_node, travel_time, detection_probability))
                    came_from[next_node] = current

    return None, None, None, None  # Return None if no path is found within the time constraint

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

import matplotlib.patches as mpatches

def visualize_path(path, node_field, arcs, ax):
    # Define colors for different modes of travel
    travel_colors = {'walking': 'red', 'crawling': 'blue', 'sneaking': 'green', 'crawling_sneaking': 'black', 'walking_sneaking': 'black'}
    for i in range(len(path) - 1):
        node_i = path[i]
        node_j = path[i + 1]
        
        # Extract the coordinates from the node_field
        x_i, y_i = node_field[node_i][0]
        z_i = node_field[node_i][1]
        
        x_j, y_j = node_field[node_j][0]
        z_j = node_field[node_j][1]
        
        # Retrieve the arc information
        arc_info = arcs.get((node_i, node_j)) or arcs.get((node_j, node_i))
        if not arc_info:
            raise ValueError(f"No arc information found for the path between {node_i} and {node_j}")
        
        # Extract the mode of travel from the arc information
        mode_of_travel = arc_info[2]  # Assuming the third element is the mode of travel
        
        # Get the color for the mode of travel
        color = travel_colors.get(mode_of_travel, 'magenta')  # Default to 'magenta' if mode of travel is not found
        
        # Plot the path segment with the determined color
        ax.plot([x_i, x_j], [y_i, y_j], [z_i, z_j], color=color, linewidth=2)


# ... (other parts of your code) ...
colors_dic={'walking': 'red', 'crawling': 'blue', 'sneaking': 'green', 'crawling_sneaking': 'black', 'walking_sneaking': 'black'}

def detection_fields_path(mode_of_travel, perpendicular, plot_node_field=False, path=None, seekers=seekers, arcs=None, colors_dic=colors_dic):
    detection_fields(mode_of_travel, perpendicular, plot_node_field)
        
    if path is not None and arcs is not None and colors_dic is not None:
        node_field = create_node_field()
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            arc_key = (node_i, node_j) if (node_i, node_j) in arcs else (node_j, node_i)
            if arc_key in arcs:
                mode_of_travel = arcs[arc_key][2]  # Assuming the third element is the mode of travel
                color = colors_dic.get(mode_of_travel, 'magenta')  # Default to 'magenta' if mode of travel is not found
                x_i, y_i = node_field[node_i][0]
                x_j, y_j = node_field[node_j][0]
                plt.plot([x_i, x_j], [y_i, y_j], color=color, linewidth=3)
    
    if plot_node_field:
        node_field = create_node_field()
        for i in range(nodes_wide * nodes_long):
            (x, y) = node_field[i + 1][0]
            plt.scatter(x, y, color="black")
    
    plt.subplots_adjust(left=0.25, right=0.8)  # Adjusting the subplot proportions
    #fig.tight_layout(rect=[0.25, 0.05, 0.8, 0.95])  # Adjust the tight layout for better visualization

    plt.show()


def a_star_time_optimized(start, goal, arcs, node_field, max_detection, desired_map_length, desired_map_width,heuristic_on=False):
    open_set = []
    heapq.heappush(open_set, (0, start, 0, 0))  # (time, current_node, cost, probability_of_detection)
    came_from = {}
    time_so_far = {start: 0}
    crawl_time_so_far = {start:0}
    sneak_time_so_far = {start:0}
    detection_so_far = {start: 0}
    cost_so_far = {start: 0}

    while open_set:
        _, current, current_cost, current_detection = heapq.heappop(open_set)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, time_so_far[goal], current_cost, current_detection

        for next_node in node_field[current][3]:  # Adjacent nodes are in the 4th position
            travel_time = time_so_far[current] + arcs[(current, next_node)][3]  # Travel time is in the 4th position
            detection_probability = detection_so_far[current] + arcs[(current, next_node)][4]  # Detection probability is in the 5th position
            #mode_of_travel = arcs[arc_key][2]
            if detection_probability <= max_detection:
                new_time = time_so_far[current] + arcs[(current, next_node)][3]  # This time we prioritize time
                if next_node not in time_so_far or new_time < time_so_far[next_node]:
                    time_so_far[next_node] = new_time
                    cost_so_far[next_node] = current_cost + arcs[(current, next_node)][4]  # Still tracking cost for reference
                    detection_so_far[next_node] = detection_probability
                    if heuristic_on==False:
                        
                        priority = new_time #+ heuristic(node_field[next_node], node_field[goal],desired_map_length,desired_map_width)
                    else:
                        priority = new_time + heuristic(node_field[next_node], node_field[goal],desired_map_length,desired_map_width)

                    #priority = new_time + heuristic(node_field[next_node], node_field[goal], desired_map_length, desired_map_width)
                    heapq.heappush(open_set, (priority, next_node, new_time, detection_probability))
                    came_from[next_node] = current

    return None, None, None, None  # Return None if no path is found within the detection constraint

import heapq
import numpy as np
import time

def show_paths():
    start_time = time.time()
    
    arcs , nf =get_arcs2() # this is the slowest part

    
    start_node = 1  # Replace with the actual start node
    goal_node = 961  # Replace with the actual goal node
    time_constraint = 10000  # Adjust as needed
    
    #path, cost, travel_time, detection_probability = a_star(start_node, goal_node, arcs, nf, time_constraint,desired_map_length,desired_map_width)
    
    least_detectable_path, cost, travel_time, detection_threshold = a_star(start_node, goal_node, arcs, nf, time_constraint, desired_map_length, desired_map_width)
    
    
    
    if least_detectable_path is not None:
        print("Path found:", least_detectable_path)
        print("Cost:", cost)
        print("Travel Time:", travel_time)
        print("Detection Probability:", detection_threshold)
        #mode_of_travel, perpendicular, plot_node_field, path=None, seekers=seekers, arcs=None, colors_dic=colors_dic
        detection_fields_path('walking', False, False, least_detectable_path,seekers,arcs,colors_dic)
    else:
        print("No path found within the time constraint.")
    
    
    
    if least_detectable_path is not None:
        quickest_path, travel_time_new, cost, new_detection_probability = a_star_time_optimized(start_node, goal_node, arcs, nf, detection_threshold, desired_map_length, desired_map_width)
    iterations=1
    while quickest_path == None:
        iterations=+1
        detection_threshold=detection_threshold + .0005
        quickest_path, travel_time_new, cost, new_detection_probability = a_star_time_optimized(start_node, goal_node, arcs, nf, detection_threshold, desired_map_length, desired_map_width)
    
    if quickest_path is not None:
        print("Path found(2nd iteration):", quickest_path)
        print("Original Travel Time:", travel_time,"new Travel Time:", travel_time_new)
        print("Original Detection Probability:", detection_threshold, "New Detection Probability:", new_detection_probability)
        print("iterations:", iterations)
        #mode_of_travel, perpendicular, plot_node_field, path=None, seekers=seekers, arcs=None, colors_dic=colors_dic
        detection_fields_path('walking', False, False, quickest_path,seekers,arcs,colors_dic)
    else:
        print("No path found within the detection constraint.")
    a_star_time = time.time() - start_time
    
    return {
        'least_detectable_path': least_detectable_path,
        'quickest_path': quickest_path,
        'cost': cost,
        'original_travel_time': travel_time,
        'new_travel_time': travel_time_new,
        'original_detection_probability': 'detection_threshold_new',
        'new_detection_probability': new_detection_probability,
        'iterations': iterations,
        'a_star_time': a_star_time,
        # 'plot_time': plot_time
    }

"""
-------------------------------------------------------------------------------
Ideas to Implement
-------------------------------------------------------------------------------
    Increase Speed    
        o Instead of a seeker box of half length z, create a seeker circle of radious z (different type of uncertainty)
            - Find true angle (theta) from seeker to evader (arctan2)
            - seeker_closest= rotate(theta) @ (seeker+(z,0))
            - seeker_left = rotate(theta) @ (seeker+(0, z))
            - seeker_right = rotate(theta) @ (seeker+(0, -z))
            - seeker_max = max elevation seeker (grid / ?polar? search in circle of uncertainty)
        o Only check three closest seekers in each seeker box
        o Analyze f(gamma) as a function of distance and angle between movement and evader to seeker
            - Potentially remove need to calculate alpha beta and gamma
        o 
"""

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
    
    N=nodes_wide*nodes_long
    robo_nodes=[i+1 for i in range(N)]
    
    "Create arcs dictionary"
    arcs={}
    arc_length=get_arc_length()/2
    
    checked=0
    for i in tqdm(range(N)):
        node_i=i+1
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        position_i=np.array(coordinate_i+(elevation_i,))
        node_i_land=classify_node(position_i)
        for node_j in adjacent_nodes_i:
            "Dont recaclulate arcs arleady found"
            key=(node_j,node_i)
            if key not in arcs:
                [coordinate_j, elevation_j, vegetation_j, adjacent_nodes_j] = node_field[node_j]
                position_j=np.array(coordinate_j+(elevation_j,))
                node_j_land=classify_node(position_j)
                
                "Only add Arc if arc is possible (no cliff and neither node is in water)"
                if abs(elevation_i-elevation_j)<=2*step_size and node_i_land and node_j_land: #WHAT IS THE STEEPEST GRADE A HUMAN CAN CLIMB
                    
                    "Determine Time for travel"
                    distance = np.linalg.norm(position_j-position_i)
                    mode_of_travel='robo'
                    
                    "Determine time base on speed, i average vegetation encountered"
                    average_vegetation=(vegetation_i+vegetation_j)/2
                    vegetation_factor=1-(2/9)*average_vegetation #NEEDS VALIDATION THROUGH TESTING!!!!!!!!!!!
                    travel_time=distance/(vegetation_factor*speed_dic[mode_of_travel])
                    
                    "Determine Probability of detection"
                    #visual_detection=get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups)
                    #audio_detection=get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups)
                    energy_level = 1
                #risk_level = max(visual_detection,audio_detection)    
                arcs[(node_i,node_j)]=[position_i, position_j, mode_of_travel, travel_time, energy_level]
                arcs[(node_j,node_i)]=[position_j, position_i, mode_of_travel, travel_time, energy_level]
                checked+=1
                # print(node_i,"-> ", node_j,"complete, risk=", np.round(risk_level,4), "Status:",np.round(100*(checked)/arc_length,2),"% Complete")
    # print("Calculation Time:",time.time()-start_time)
    return arcs, node_field

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

    node_id=1
    for l in range(nodes_long):
        for w in range(nodes_wide):
            coordinate = (w*step_size,l*step_size)
            (x,y)=(coordinate[0]*map_width_scale,coordinate[1]*map_length_scale)
            elevation = (elevation_map.getpixel((x, y))[0]/255)*max_elevation
            r= vegetation_map.getpixel((x, y))[0] 
            vegetation = (3-(3*(r/255))) #vegetation is scaled "continuously" from 0 (None) to 3 (Dense)
            #robo level
            node_field[node_id] = [coordinate, elevation, vegetation, get_adjacent_nodes(node_id, coordinate)]
            node_id+=1

    return node_field