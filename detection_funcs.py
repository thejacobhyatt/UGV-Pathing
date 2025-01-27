import numpy as np
import math

seeker_orientation_uncertainty = {'human': (62*np.pi/180)+np.pi/2, 'bunker': 62*np.pi/180} #Enemy capabilities will influence this (among other things), will likely need to expand dictionary and definitions for each key
point_source_dic = {'charged': np.array([20/3, 10]), 'charging':np.array([40/3, 30])}
speed_dic = {'charged':1, 'charging':1} #VALIDATE THESE SPEEDS THROUGH TESTING!
height_dic = {'charged':1.5, 'charging':1.5} #CONFIRM HEIGHTS - ASSUME 

def get_audio_detection(position_self, position_neighbor, mode_of_travel, seeker_groups, vegetation_map, distance_scale):
    x1 , y1, e1 = position_self 
    x2 , y2, e2 = position_neighbor

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    
    audio_detection = []
    vegetation = (vegetation_map[int(y), int(x), 0] / 255)
    
    point_source = np.array([vegetation,1])@point_source_dic[mode_of_travel] - 50 # background noise = 50
    
    for worst_case_seekers in seeker_groups:
        for seeker in seeker_groups[worst_case_seekers]:

            [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
            seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
            seeker_to_evader = position_self-seeker_coord
            
            distance1 = np.linalg.norm(seeker_to_evader) * distance_scale
            distance2 = np.linalg.norm(position_neighbor-seeker_coord) * distance_scale
            a = (probability_audio(point_source, distance1) + probability_audio(point_source, distance2)) / 2
            audio_detection.append(a)
    
    return max(audio_detection)

def probability_audio(point_source, dist):
    dist = dist
    deltaDB = point_source - 2*np.log(dist)
    k = .1
    
    return 1 / (1 + np.exp(-k*deltaDB))


def get_seeker_group(templated_seeker, elevation_map, MAX_ELEVATION):
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
        seeker_group.append([x, y, (elevation_map[int(y), int(x), 0] / 255)*MAX_ELEVATION+2, orient_left, orient_right])
        
    [x_center,y_center]=seeker_loc
    locations=[[(x_center-loc_uncertainty+i,y_center-loc_uncertainty+j) for i in range(2*loc_uncertainty)] for j in range(2*loc_uncertainty)]

    "Find location of highest seeker in seeker box and add to seeker box"
    e_max=-999
    for i in range(2*loc_uncertainty):
        for j in range(2*loc_uncertainty):
            [x2,y2] = locations[i][j]
            e=(elevation_map[int(y), int(x), 0] / 255)*MAX_ELEVATION
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

def get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups, seekers, elevation_map, max_elevation, vegetation_map, distance_scale):
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
        distance_position_i=np.linalg.norm(np.array(seeker_coord)-position_i[:2]) * distance_scale
        distance_position_j=np.linalg.norm(np.array(seeker_coord)-position_j[:2]) * distance_scale
        if distance_position_i <=z or distance_position_j <= z:
            return 1

    visual_detection=[]
    for worst_case_seekers in seeker_groups:

        for seeker in seeker_groups[worst_case_seekers]:
            los_i = get_los(seeker, position_i, elevation_map, max_elevation,vegetation_map,distance_scale) #Line of Sight to start postion
            los_j = get_los(seeker, position_j, elevation_map, max_elevation, vegetation_map,distance_scale) #Line of sight to stop position
            los = (los_i+los_j)/2 #average line of sight from start to stop
            
            if los==0:
                visual_detection.append(0)
            else:
                [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
                seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
                alpha = get_alpha(seeker_coord, position_i, position_j, speed_dic[mode_of_travel])
                beta = get_beta(seeker_coord, position_i, position_j, height_dic[mode_of_travel], distance_scale)
                trace_ratio = closed_form_ratio(alpha, beta)
                # detection_probability = 999*trace_ratio/(998*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = 101*trace_ratio/(100*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = detection_over_step(int(travel_time), detection_probability) #testing this function out
                visual_detection.append(los*detection_probability)
    return max(visual_detection)

def get_los(seeker, evader_loc, elevation_map, max_elevation,vegetation_map,distance_scale):
    if evader_in_blindspot(seeker, evader_loc):
        # print("blindspot")
        return 0
    [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
    seeker_loc=np.array([seeker_x, seeker_y, seeker_elevation])    
    
    "Create vector function r(t) from seeker to evader (the seeker's line of sight)"
    r0=seeker_loc[:]
    v=evader_loc-seeker_loc
    distance=np.linalg.norm(v) * distance_scale
    t=np.linspace(0,1,int(distance/(10/2.5))) #searches along route every ~step_size/2.5 meters
    r=[r0+v*t[i] for i in range(len(t))]
    
    "Find the vegetation factor on visibility"
    vegetation_factor=1
    for position in r:
        [x,y,e]=position
        ground_elevation = (elevation_map[int(y), int(x), 0] / 255)*max_elevation
        if ground_elevation>e:
            "Line of sight blocked by obstace: Evader is in deadspace"
            # print("deadspace")
            return 0
        r = vegetation_map[int(y), int(x), 0]
        vegetation = (3-(3*(r/255)))
        vegetation_factor *= (1-(1/30)*vegetation) #assumes linear probability of seeing through vegetation probability = 1-(2/30)*density at any given point.
        #instances of seeing through vegetatin are independent, thus the probability of seeing through a bunch of vegetation of their multiplication.
        if vegetation_factor<.01:
            # print("VEGETATED")
            return 0
    return vegetation_factor

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


def get_beta(seeker_node, start_node, end_node, height, distance_scale):
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
    distances=[np.linalg.norm(seeker_node-start_node), np.linalg.norm(seeker_node-end_node)] * distance_scale
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