# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:36:15 2024

@author: robert.jane
"""

# solve in gurobi 


##Import Python Distribution Modules
import os
import sys
import math
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Add System Path
sys.path.insert(1, 'Code/')

#Import Python Custom Moduels
from exenf_alog import direction_of_travel
from exenf_alog import exenf_cost
    
#Start Time
start_time = time.time()

#Initial/Final Point (m)
p1 = [00.00,00.00,+00.00]
p2 = [10.00,10.00,-40.00]

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

#Heading Direction (deg)
heading = direction_of_travel(p1,p2,math)

#Debug
debug = False

#Input Parameters
params = [p1, p2, travel_time, platform_name, added_mass, wind_velocity, wind_direction, heading, debug]

#Input Function Handles
fcns = [np, minimize, interp1d, math, os]

#Exergy/Energy Cost
Jcon, Jgen, msg = exenf_cost(params,fcns)

#Execution Time
# print("--- %s seconds ---" % (time.time() - start_time))