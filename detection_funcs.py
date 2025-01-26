import numpy as np

seeker_orientation_uncertainty = {'human': (62*np.pi/180)+np.pi/2, 'bunker': 62*np.pi/180} #Enemy capabilities will influence this (among other things), will likely need to expand dictionary and definitions for each key
point_source_dic = {'charged': np.array([20/3, 60]), 'charging':np.array([40/3, 90])}


def get_audio_detection(position_self, position_neighbor, mode_of_travel, seeker_groups, vegetation_map):
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
            
            distance1 = np.linalg.norm(seeker_to_evader)
            distance2 = np.linalg.norm(position_neighbor-seeker_coord)
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