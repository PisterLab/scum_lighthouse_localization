import numpy as np 
import pandas as pd
import sys
from StateEstimator import UKF6DoF, Kalman
import matplotlib.pyplot as plt
from madgwickahrs import MadgwickAHRS
from py3dmath import Vec3, Rotation
from quaternion import Quaternion
import Lighthouse as lh
from Optitrack import load_optitrack, plot_truth_traj, project_truth_to_az_el
from processData import generate_imu

class Lighthouse:
    def __init__(self,x,y,z,theta):
        
        self.pose = np.array([[1,0,0], 
                     [0,np.cos(theta), np.sin(theta)],
                     [0,-np.sin(theta), np.cos(theta)]])

        self.translation = np.array([[1,0,0,-x],
                                     [0,1,0,-y],
                                     [0,0,1,-z]])
        self.K = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])

        self.P = np.matmul(np.matmul(self.K,self.pose),self.translation)

#load imu lighthouse data
def load_data(take_name):

    imu_filename = "lighthouse_imu_data/" + take_name + "_imu.csv"
    lighthouse_filename = "lighthouse_imu_data/" + take_name + '_lighthouse.csv' 


    #load imu data
    with open(imu_filename) as file:
        column_names = ['accel_x','accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z','timestamp']
        imu_df = pd.read_csv(file, names = column_names)
        imu_df['timestamp'] = imu_df['timestamp']/10e6

    #load lighthouse data
    with open(lighthouse_filename) as file:
        column_names = ['azA', 'elA', 'azB', 'elB','timestamp']
        lighthouse_df = pd.read_csv(file, names = column_names)
        lighthouse_df['timestamp'] = lighthouse_df['timestamp']/10e6

    return imu_df,lighthouse_df 




if __name__ == "__main__":

    take_name = sys.argv[1]

    #read data
    print("Loading Experiment: ", sys.argv[1])
    imu_df , lighthouse_d = load_data(sys.argv[1])

    #load lighthouse data
    lighthouse_filename = "lighthouse_imu_data/" + take_name + '_lighthouse.csv' 
    lighthouse_df = lh.load_lh(lighthouse_filename,take_name, align = True)

    #load imu data
    imu_filename = "lighthouse_imu_data/" + take_name + "_imu.csv"
    imu_df = lh.load_imu(imu_filename,take_name,align = True)

    #load approximate ligthouse locations from old experiments
    with open("gnd_to_lh1_" + 'take5' + ".npy",'rb') as f:
        gnd_to_lh1_pose = np.load(f)

    with open("gnd_to_lh2_" + 'take5' + ".npy",'rb') as f:
        gnd_to_lh2_pose = np.load(f)

    lh1 = np.load("lh1_gnd_" + 'take5' + ".npy")
    lh2 = np.load("lh2_gnd_" + 'take5' + ".npy")

    #setup lighthouse locations
    lighthouse1 = lh.Lighthouse(x= lh1[0], y =lh1[1], z = lh1[2], pose = np.linalg.inv(gnd_to_lh1_pose))
    lighthouse2 = lh.Lighthouse(x= lh2[0], y =lh2[1], z = lh2[2], pose = np.linalg.inv(gnd_to_lh2_pose))

    #get opititrack data
    filename = "lighthouse_imu_data/" + sys.argv[1] + '_optitrack.csv' 
    gnd_df = load_optitrack(filename)
    
    #project optitrack data to azimuth and elevation measurements
    azimuth1_gnd, elevation1_gnd, azimuth2_gnd, elevation2_gnd = project_truth_to_az_el(scum_df = gnd_df,
                                                                                        lh1 = lh1,
                                                                                        lh2 = lh2,
                                                                                        lh1_to_gnd = np.linalg.inv(gnd_to_lh1_pose), 
                                                                                        lh2_to_gnd = np.linalg.inv(gnd_to_lh2_pose))

    #plot azimuth and elevation data versus lighthouse data 
    #lh.plot_raw_data(azimuth1_gnd, azimuth2_gnd, elevation1_gnd, elevation2_gnd, lighthouse_df, gnd_df, show = False)

    #generate cam image values
    cam_points = lh.generate_cam_points(lighthouse_df)

    #calibrate
    lh1_calibrated, lh2_calibrated, lh1_gnd_proj, lh2_gnd_proj = lh.calibrate(gnd_df,lighthouse1.P,lighthouse2.P,cam_points,plot_results = False)

    #generate lighthouse tracked trajectory
    trajectory = lh.triangulate_scum(lighthouse_df,lh1_calibrated, lh2_calibrated)

    #plot lighthouse trajectory
    lh.plot_trajectories(trajectory,gnd_df,show =False)

    #get derived synthetic values
    gnd_time, velocity, acceleration, compass, omega, rotations, eulers,position = generate_imu(gnd_df,xl_noise = 3.6e-3*9.8*10)

    print(len(rotations))

    #feed data into state estimator object
    Q_6 = np.diag([1,1,1,1,1,1,.1,.1,.1])*0.01
    xm_0_6 = np.array([0,0,0,0,0,0,0,0,0])
    Pm_0_6 = np.diag([1,1,1,1,1,1,1,1,1])*0.01
    R_ukf = 0.1

    start = 300

    print(imu_df)
    #create ukf estimator
    #TODO also use the 3dof estimator with gravity removed
    '''
    estimator = UKF6DoF(Q=Q_6, R = R_ukf, xm_0 = xm_0_6, Pm_0 = Pm_0_6, time= imu_df['timestamp_optitrack'], R_eo = np.eye(3), 
                        gnd_to_lh1_pose = lighthouse1.pose, gnd_to_lh2_pose = lighthouse2.pose, 
                        R_init = np.eye(3) , t0 = imu_df['timestamp_optitrack'].iloc[start],Rg = 20,gravity = False)
    '''
    Q = np.diag([1,1,1,0.1,0.1,0.1])*0.01
    R = np.diag([.1,.1,.1])*0.01
    xm_0 = np.array([0,0,0,0,0,0])
    Pm_0 = np.diag([1,1,1,1,1,1])*0.01

    dummy_times = np.arange(imu_df['timestamp_optitrack'].iloc[start],imu_df['timestamp_optitrack'].iloc[len(imu_df)-1],1/100)

    estimator = Kalman(Q=Q, R = R, xm_0 = xm_0, Pm_0 = Pm_0, time= imu_df['timestamp_optitrack'], R_eo = np.eye(3), thresh = 100,t0 = imu_df['timestamp_optitrack'].iloc[start])
    #setup inital rotation for madgwick filter
    initial_rot = Rotation(1,0,0,0).from_euler_YPR([0,.3,2.8])
    initial_quat = Quaternion(initial_rot.q[0],initial_rot.q[1],initial_rot.q[2],initial_rot.q[3])
    comp_filt = MadgwickAHRS(1/60,quaternion = initial_quat, beta = 1)
    comp_eulers = []
    rot_last = None
    #start estimation loop 
    for i in range(start,len(imu_df)):
        #get current imu point
        imu_point = imu_df.iloc[i]
        acceleration = imu_point[['accel_x', 'accel_y', 'accel_z']].values / 16384 * 9.8 
        omega = imu_point[['gyro_x' , 'gyro_y', 'gyro_z']].values / 65535 *2*250 *np.pi/180
        time = imu_point['timestamp_optitrack']




        if time > 0:

            if estimator.is_valid_time(time):

                #madgwick attitude estimator
                comp_filt.update_imu(omega,acceleration)
                comp_quat = comp_filt.quaternion
                comp_rot = Rotation(comp_quat[0],comp_quat[1],comp_quat[2],comp_quat[3]) #sensor to earth
                humcomp_rot = Rotation(1,0,0,0).from_rotation_matrix(np.linalg.inv(comp_rot.to_rotation_matrix())) #earth to sensor
                comp_eulers.append(comp_rot.to_euler_YPR())
                rotated_grav = comp_rot.to_rotation_matrix()@ np.array([0,0,9.8]).reshape((3,1))

                #predict step
                rot_idx = np.searchsorted(gnd_time,time)
                if rot_idx < len(rotations):
                    rot_mat = rotations[rot_idx].to_rotation_matrix()
                else:
                    rot_mat = rot_last

                earth_accel = rot_mat @ acceleration - np.array([0,-9.8,0])
                print('accel', earth_accel)
                estimator.predict(earth_accel.reshape((3,1)),omega = None,time= time)
                #s_priors6 = estimator.predict(acceleration.reshape((3,1)) - rotated_grav ,omega,time)

                #get current measurement
                measurement = estimator.meas_available(time,trajectory)
                #measurement = estimator.meas_available_azel(imu_point['timestamp_optitrack'],lighthouse_df)

                #update step
                estimator.update(measurement)
                #estimator.update(measurement,s_priors6,acceleration)
                rot_last = rot_mat
    print( estimator.get_states())
    estimator.plot_tracking(gnd_df,show = True)

    comp_eulers = np.array(comp_eulers)
    plt.figure()
    plt.plot(imu_df['timestamp_optitrack'])
    print(comp_eulers)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(estimator.get_eulers()[:,0])
    plt.plot(comp_eulers[:,0])
    plt.title('Attitude Tracking')
    plt.ylabel('Yaw')
    plt.legend(['MUKF', 'Madgwick'])

    plt.subplot(3,1,2)
    plt.plot(estimator.get_eulers()[:,1])
    plt.plot(comp_eulers[:,1])
    plt.ylabel('Pitch')

    plt.subplot(3,1,3)
    plt.plot(estimator.get_eulers()[:,2])
    plt.plot(comp_eulers[:,2])
    plt.ylabel('Roll')

    estimator.plot_states()
    #plot
    plt.figure()
    plt.subplot()
    plt.plot(estimator.xm[0,:])
    plt.show()