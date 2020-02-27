import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_optitrack(optitrack_filename,body_name = 'cx7'):
    with open(optitrack_filename) as file:
        time_name = ['Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)']

        ground_truth_df = pd.read_csv(file, header = [2,4,5])
        scum_gnd_df = ground_truth_df[body_name]
        gnd_time_df = ground_truth_df['Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)']
        scum_gnd_df['Time','Time'] = gnd_time_df
        #TODO: rotate orientation and position by optitrack to earth frame 

    return scum_gnd_df

def plot_truth_traj(scum_df,VICON_SCALE = 1):
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(scum_df['Time', 'Time'],scum_df['Position','X']/VICON_SCALE)
    plt.title('Ground Truth X')

    plt.subplot(3,1,2)
    plt.plot(scum_df['Time', 'Time'],scum_df['Position','Y']/VICON_SCALE)
    plt.title('Ground Truth Y')

    plt.subplot(3,1,3)
    plt.plot(scum_df['Time', 'Time'],scum_df['Position','Z']/VICON_SCALE)
    plt.title('Ground Truth Z')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Ground Truth Trajectory')
    ax.plot(xs = scum_df['Position','X']/VICON_SCALE, ys = scum_df['Position','Y']/VICON_SCALE, zs = scum_df['Position','Z']/VICON_SCALE)
    #ax.scatter(xs = [lh1[1][0], lh2[1][0]], 
    #            ys = [lh1[1][1], lh2[1][1]], 
    #            zs = [lh1[1][2], lh2[1][2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('scaled')

#this function will project the ground truth position to azimuth and elevation angles relative to the lighthouses. 
def project_truth_to_az_el(scum_df,lh1,lh2,lh1_to_gnd,lh2_to_gnd):

    scum_pos = scum_df['Position']
    #rotate ground truth xyz to lh1 coordinates

    traj_lh1 = np.dot(np.linalg.inv(lh1_to_gnd),(scum_pos[['X','Y','Z']] - lh1).T)

    #print('######## Trajectory in LH 1 Coordinate Frame')
    #print(traj_lh1)
    #find azimuth and elevation using atan2
    elevation1 = np.arctan2(traj_lh1[1,:],traj_lh1[2,:])
    azimuth1 = np.arctan2(traj_lh1[0,:],traj_lh1[2,:])


    #rotate ground truth xyz to lh1 coordinates
    traj_lh2 = np.dot(np.linalg.inv(lh2_to_gnd),(scum_pos[['X','Y','Z']] - lh2).T)

    #find azimuth and elevation using atan2
    elevation2 = np.arctan2(traj_lh2[1,:],traj_lh2[2,:])
    azimuth2 = np.arctan2(traj_lh2[0,:],traj_lh2[2,:])

    return azimuth1, elevation1, azimuth2, elevation2 