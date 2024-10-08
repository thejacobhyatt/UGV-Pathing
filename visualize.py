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

"""
-------------------------------------------------------------------------------
Universals
-------------------------------------------------------------------------------
"""
speed_dic = {'robo':1} #VALIDATE THESE SPEEDS THROUGH TESTING!
height_dic = {'robo':1.5} #CONFIRM HEIGHTS - ASSUME 
transition_time_dic = {'crawling_sneaking': 2, 'walking_sneaking': .5} #CONFIRM THESE TIMES IN SECONDS!
seeker_orientation_uncertainty = {'human': (62*np.pi/180)+np.pi/2, 'bunker': 62*np.pi/180} #Enemy capabilities will influence this (among other things), will likely need to expand dictionary and definitions for each key
point_source_dic ={'walking' : np.array([40/3, 20]), 'sneaking' : np.array([30/3, 15]), 'crawling' : np.array([20/3, 10]), 'crawling_sneaking':np.array([30/3, 15]), 'robo' : np.array([40/3, 20])}
"""
-------------------------------------------------------------------------------
User Input
-------------------------------------------------------------------------------
"""
#Relative PATH in GitHub Project
file_path = '..\\OTP Figures\\'
file_name=file_path+"Buckner" #map identifier
desired_lower_left_corner = (200, 200) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
desired_upper_right_corner = (400, 400) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
step_size=10  #desired distance, in meters, between nodes CANNOT BE TOO LARGE OR WILL CAUSE OverflowError when determining probability
# seekers ={1: [(5,5), 5, 0, seeker_orientation_uncertainty['human']]}
seekers={1 : [(50,50), 15, 0, seeker_orientation_uncertainty['human']], 2 : [(100,150), 15, -np.pi/2, seeker_orientation_uncertainty['human']], 3 : [(150,50), 10, 3*np.pi/4, seeker_orientation_uncertainty['bunker']]}
# #{seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ], next seeker : [...], ...}
fog_coef = 0

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

