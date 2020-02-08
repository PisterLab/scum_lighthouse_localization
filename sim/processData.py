import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from py3dmath import Vec3, Rotation
import skinematics as skin
from skinematics.sensors.manual import MyOwnSensor
import pickle
from StateEstimator import Kalman, UKF
lh_index = 0

#this function will align the lighthouse data to the vicon data. It will
#return lighthouse data timestamp column that is temporally aligned with respect to the opititrack data
def align_data(lh_data,optitrack_data = None):

    #time offsets (lh - optitrack)
    offsets = {'take1' : 284.6-85.4039,
                'take2' : 257.359 - 2.94922,
                'take3' : 58.08 - 5.671,
                'take4' : 150.123 - 96.4074,
                'take5' : 116.671-84.6218}

    aligned_timestamps = lh_data['timestamp_seconds'] - offsets[sys.argv[1]] 
    return aligned_timestamps


#processes optitrack data with the filename string. Returns
#a pandas dataframe. 
def load_data(take_name):

    optitrack_filename = take_name + "_optitrack.csv"
    lighthouse_traj_filename = 'lighthouse_traj_measurements_' + take_name + '.npy' 


    #load ground truth file
    with open(optitrack_filename) as file:
        time_name = ['Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)']

        ground_truth_df = pd.read_csv(file, header = [2,4,5])

        scum_gnd_df = ground_truth_df['scum']
        gnd_time_df = ground_truth_df['Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)']
        scum_gnd_df['Time','Time'] = gnd_time_df

    #load lighthouse trajectory measurements
    lighthouse_trajectory = np.load(lighthouse_traj_filename)

    return scum_gnd_df,lighthouse_trajectory 

def generate_imu(truth_df,xl_noise,gyro_noise = 0.01):
    #3.6 mg noise 
    #earth to optitrack

    roll = -np.pi/2
    pitch = np.pi
    R_eo_rot = Rotation.from_euler_YPR([0,pitch,roll])
    R_eo = np.array(R_eo_rot.to_rotation_matrix())
    print(R_eo)

    #extract time
    time = truth_df['Time']['Time'].values

    #extract position
    position = np.transpose(np.linalg.inv(R_eo) @ truth_df['Position'].values.T)

    #extract rotation
    quats = truth_df['Rotation'].values
    rotations = []
    last_valid = []
    for quat in quats:
        if np.isnan(quat[2]):
            quat = last_valid
        else:
            last_valid = quat

        rotations.append(Rotation(quat[3],quat[0],quat[1],quat[2]))

    #generate body frame acceleration
    velocity = np.zeros( (len(time),3) )
    acceleration = np.zeros( (len(time),3) )
    omega = np.zeros( (len(time),3) )
    compass = np.zeros( (len(time),3) )
    eulers = np.zeros( (len(time),3) )

    for i in range(1,len(time)):

        delta_t = time[i] - time[i-1]
        delta_pos = (position[i,:] - position[i-1,:])
        Ri = rotations[i].to_rotation_matrix()
        velocity[i,:] = Ri @ np.matmul(R_eo,delta_pos) / delta_t
        acceleration[i,:] = (velocity[i,:] - velocity[i-1,:]) / delta_t + (np.random.rand()-0.5) * xl_noise #+ Ri @ (R_eo @ np.array([0,0,9.8]))
        compass[i,:] = Ri @ (R_eo @ np.array([1,0,0]))

        rot_curr = rotations[i].to_rotation_vector()
        rot_prev = rotations[i-1].to_rotation_vector()
        d_rot = (rot_curr - rot_prev)/(delta_t)
        rot_cross = rotations[i].to_rotation_vector().to_cross_product_matrix().T

        omega[i,:] = np.squeeze((np.linalg.inv(np.eye(3) - 1/2*rot_cross) * d_rot).to_array()) + (np.random.rand(3) - 0.5) * gyro_noise
        eulers[i,:] = rotations[i].to_euler_YPR() 


    return time, np.nan_to_num(velocity), np.nan_to_num(acceleration), compass, omega, rotations, eulers, position



#r is gnd to lighthouse
def linearize_meas(xp,angle,xl,R,meas_type):

    #elevation (angle = atan(y/z))
    x = R[0,:] @ (xp[0:3] - xl)
    y = R[1,:] @ (xp[0:3] - xl)
    z = R[2,:] @ (xp[0:3] - xl)
    print('xp: ', xp)
    print('xl:', xl)
    print("x with respect to lighthouse: ", x,y,z)
    assert(meas_type == 'el' or meas_type == 'az')

    if meas_type == 'el':
        #elevation (angle = atan(y/z))
        dh_dx = 1 / (1+(y/z)**2) * (z * R[1,0] - y * R[2,0])/(z**2)

        dh_dy = 1 / (1+(y/z)**2) * (z * R[1,1] - y * R[2,1])/(z**2)

        dh_dz = 1 / (1+(y/z)**2) * (z * R[1,2] - y * R[2,2])/(z**2)

    else:
        #azimuth(angle = atan(x/z)))
        dh_dx = 1 / (1+(x/z)**2) * (z * R[1,0] - x * R[2,0])/(z**2)

        dh_dy = 1 / (1+(x/z)**2) * (z * R[1,1] - x * R[2,1])/(z**2)

        dh_dz = 1 / (1+(x/z)**2) * (z * R[1,2] - x * R[2,2])/(z**2)

    H = np.array([dh_dx, dh_dy, dh_dz, 0, 0, 0])
    print('H: ', H)
    return H

def meas_available_azel(time,lighthouse_df,lh1_gnd, lh2_gnd):
    global lh_index

    while(lighthouse_df['timestamp_optitrack'].iloc[lh_index] - time < 0):
        lh_index += 1

    if lighthouse_df['timestamp_optitrack'].iloc[lh_index] - time <= 1/60:
        
        if lighthouse_df['azA'].iloc[lh_index] > 0:
            #calculate relative
            measurement = {'z': np.deg2rad(lighthouse_df['azA'].iloc[lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : lh1_gnd,'lh_num': 1}
             
        if lighthouse_df['azB'].iloc[lh_index] > 0:
            measurement = {'z': np.deg2rad(lighthouse_df['azB'].iloc[lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : lh2_gnd,'lh_num': 2}

        if lighthouse_df['elA'].iloc[lh_index] > 0:
            measurement = {'z': np.deg2rad(lighthouse_df['elA'].iloc[lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : lh1_gnd,'lh_num': 1}

        if lighthouse_df['elB'].iloc[lh_index] > 0:
            measurement = {'z': np.deg2rad(lighthouse_df['elB'].iloc[lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : lh2_gnd,'lh_num': 2}

        lh_index += 1

        return measurement

    else:
        return -1 

def exp_meas(xp,meas_type,lh_loc, R):
    print('expected measure diff: ',R @ (xp[0:3] - lh_loc))
    if meas_type == 'az':
        xp_lh = R @ (xp[0:3] - lh_loc)
        angle = np.arctan2(xp_lh[0],xp_lh[2]) 
    else:
        xp_lh = R @ (xp[0:3] - lh_loc)
        angle = np.arctan2(xp_lh[1],xp_lh[2]) 

    return angle 

if __name__ == "__main__":

    print("Loading Experiment: ", sys.argv[1])

    lighthouse_filename = sys.argv[1] + "_lighthouse.csv"

    #load lighthouse data file
    with open(lighthouse_filename) as file:
        if sys.argv[1] == 'take1' or sys.argv[1] == 'take5':
            lighthouse_df = pd.read_csv(file, header=None, names = ['azA', 'elA', 'azB', 'elB', 'timestamp_10.82 Hz'], dtype = float)
        else: 
            lighthouse_df = pd.read_csv(file, header=None,lineterminator = ']', sep = ',', names = ['azA', 'elA', 'azB', 'elB', 'timestamp_10.82 Hz'],dtype = float)

        lighthouse_df['timestamp_seconds'] = lighthouse_df['timestamp_10.82 Hz'] / 10.82e6
        lighthouse_df['timestamp_optitrack'] = align_data(lighthouse_df)

    #####

    truth, lh_traj = load_data(sys.argv[1])
    with open("aligned_lighthouse_" + sys.argv[1] + ".pickle",'rb') as f:
        az_el = pickle.load(f)

    with open("gnd_to_lh1_" + sys.argv[1] + ".npy",'rb') as f:
        gnd_to_lh1_pose = np.load(f)

    with open("gnd_to_lh2_" + sys.argv[1] + ".npy",'rb') as f:
        gnd_to_lh2_pose = np.load(f)

    lh1 = np.load("lh1_gnd_" + sys.argv[1] + ".npy")
    lh2 = np.load("lh2_gnd_" + sys.argv[1] + ".npy")

    print(az_el.keys)
    for key,value in az_el.items():
        print(key)
    print(az_el['azA']['lh'])

    #generate imu data
    time, velocity, acceleration, compass, omega, rotations, eulers,position = generate_imu(truth,xl_noise = 3.6e-3*9.8*10)

    #filtering, madgwick and kalman position updates 
    data = {'rate': 60,
            'acc' : acceleration,
            'omega': omega,
            'mag': compass}

    #create object
    R_init = Rotation(0,0,0,0).from_euler_YPR((0,0,0)).to_rotation_matrix()
    imu = MyOwnSensor(in_data = data)
    imu.set_qtype('analytical') 
    print(imu.quat)
    imu.calc_position()
    print(imu.pos)
    euler_est = np.zeros((len(imu.quat),3))
    i = 0

    roll = -np.pi/2
    pitch = np.pi
    R_eo_rot = Rotation.from_euler_YPR([0,pitch,roll])
    R_eo = np.array(R_eo_rot.to_rotation_matrix())

    for quat in imu.quat:
        rotation = Rotation(quat[0],quat[1],quat[2],quat[3])
        rot_opt = Rotation(0,0,0,0).from_rotation_matrix(np.linalg.inv(R_eo) @ rotation.to_rotation_matrix() @ R_init  )
        #print(rotation.to_euler_YPR())
        euler_est[i,:] = rot_opt.to_euler_YPR()

        i+=1


    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(euler_est[:,0])
    plt.plot(eulers[:,0])

    plt.subplot(3,1,2)
    plt.plot(euler_est[:,1])
    plt.plot(eulers[:,1])

    plt.subplot(3,1,3)
    plt.plot(euler_est[:,2])
    plt.plot(eulers[:,2])

    plt.show()


    #kalman dynamics matrix linear
    A = np.block([[np.eye(3),1/60 * np.eye(3)],
                  [np.zeros((3,3)), np.eye(3)]
                  ])
    Q = np.diag([1,1,1,1,1,1])*0.001

    H = np.array([[1,0,0,0,0,0],
                 [0,1,0,0,0,0],
                 [0,0,1,0,0,0]])

    R_ukf = 0.001
    R = np.diag([.1,.1,.1])
    #e to optritrack
    roll = -np.pi/2
    pitch = np.pi
    R_eo_rot = Rotation.from_euler_YPR([0,pitch,roll])
    R_eo = np.array(R_eo_rot.to_rotation_matrix())
    xm_0 = np.array([position[1,0], position[1,1], position[1,2], velocity[1,0], velocity[1,1], velocity[1,2]])
    Pm_0 = np.diag([1,1,1,1,1,1])*0.01
    #print(A)

    estimator = Kalman(Q=Q, R = R, xm_0 = xm_0, Pm_0 = Pm_0, time= time, R_eo = R_eo)
    ukf = UKF(Q=Q, R = R_ukf, xm_0 = xm_0, Pm_0 = Pm_0, time= time, R_eo = R_eo, gnd_to_lh1_pose = gnd_to_lh1_pose, gnd_to_lh2_pose = gnd_to_lh2_pose)
    
    for i in range(1,len(time)-1):
        #predict from past state using rotation and acceleration in body axis
        estimator.predict(i,rotations,acceleration)
        s_priors = ukf.predict(i,rotations,acceleration)

        measurement = estimator.meas_available(time[i],lh_traj)
        estimator.update(measurement,i) 

        measurement = ukf.meas_available_azel(time[i],lighthouse_df, np.linalg.inv(R_eo) @ lh1, np.linalg.inv(R_eo) @ lh2)
        ukf.update(measurement,i,s_priors)

    xm = estimator.xm
    mahals = ukf.mahals
    '''
    xm = np.zeros((6,len(time)))
    xp = np.zeros((6,len(time)))
    Pp = np.zeros((6,6,len(time)))
    Pm = np.zeros((6,6,len(time)))

    Pm[:,:,0] = np.diag([1,1,1,1,1,1])
    xm[:,0] = np.array([position[1,0], position[1,1], position[1,2], velocity[1,0], velocity[1,1], velocity[1,2]])
    mahals = []
    projected_meas = []
    actual_meas = []

    for i in range(1,len(time)-1):
        
        #azel_meas = meas_available_azel(time[i],lighthouse_df, np.linalg.inv(R_eo) @ lh1, np.linalg.inv(R_eo) @ lh2)
        #print(time[i],azel_meas)
        
        
        if azel_meas != -1:
            if azel_meas['lh_num'] == 1:
                gnd_to_lh = (gnd_to_lh1_pose)
            elif azel_meas['lh_num'] == 2:
                gnd_to_lh = (gnd_to_lh2_pose)

            R = 0.00001

            H = linearize_meas(position[i,:],azel_meas['z'],azel_meas['lh_pos'],gnd_to_lh @ R_eo,azel_meas['type'])

            S = H @ Pp[:,:,i] @ H.T + R
            K = Pp[:,:,i] @ H.T / S  
            print("K", K)  
            #if i == 5010:
             #   assert(False)
            #calculate measurement's mahalabonis distance from expected measurement distribution
            #generate expected measurement
            z_tild = exp_meas(position[i,:],azel_meas['type'],azel_meas['lh_pos'],gnd_to_lh @ R_eo)
            euc_diff = azel_meas['z'] - z_tild
            print('predicted, actual: ',z_tild,azel_meas['z'])
            projected_meas.append(z_tild)
            actual_meas.append(azel_meas['z'])
            #assert(False)
            
            mahal = np.sqrt(euc_diff * 1/S * euc_diff)
            mahals.append(mahal)

            if mahal >0:
                xm[:,i] = xp[:,i] + K * ( azel_meas['z'] - z_tild) 
                Pm[:,:,i] = (np.eye(6) - K*H) @ Pp[:,:,i]
            else:
                Pm[:,:,i] = Pp[:,:,i]
                xm[:,i] = xp[:,i]
        else:
            Pm[:,:,i] = Pp[:,:,i]
            xm[:,i] = xp[:,i]
        
        
        
'''
    #
    #plt.figure()
    #plt.plot(projected_meas)
    #plt.plot(actual_meas)
    plt.figure()
    plt.hist(mahals,bins = 50)
    #plt.figure()
    #plt.plot(lh_traj[:,0],lh_traj[:,3])
    #plt.plot(time,position[:,1])

    print(velocity[1749:1755])
    plt.figure()
    plt.plot(acceleration)

    plt.figure()

    plt.plot(time,xm[0,:])
    plt.plot(time,position[:,0])
    plt.ylim([-2,0])
    plt.xlim([80,120])
    plt.scatter(lh_traj[:,0],-lh_traj[:,1],s=1,c = 'r')
    plt.ylabel('X Position (m)')
    plt.legend(['Estimated Trajectory', 'Actual Trajectory', 'Triangulated Trajectory'])

    plt.figure()

    plt.plot(time,xm[0,:])
    plt.plot(time,position[:,0])
    plt.ylim([-2,2])
    plt.xlim([80,120])
    plt.scatter(lh_traj[:,0],-lh_traj[:,1],s=1,c = 'r')
    plt.ylabel('X Position (m)')
    plt.legend(['Estimated Trajectory', 'Actual Trajectory', 'Triangulated Trajectory'])


    plt.figure()

    plt.subplot(311)
    plt.plot(time,xm[0,:])
    plt.plot(time,position[:,0])
    plt.ylim([-2,0])
    plt.xlim([80,120])
    plt.scatter(lh_traj[:,0],-lh_traj[:,1],s=1,c = 'r')
    plt.ylabel('X Position (m)')
    plt.legend(['Estimated Trajectory', 'Actual Trajectory', 'Triangulated Trajectory'])


    plt.subplot(312)
    plt.plot(time,xm[1,:])
    plt.plot(time,position[:,1])
    plt.ylim([-2,2])
    #plt.ylim([-20,20])
    plt.scatter(lh_traj[:,0],lh_traj[:,3],s=1,c = 'r')
    plt.xlim([80,120])
    plt.ylabel('X Position (m)')


    plt.subplot(313)
    plt.plot(time,xm[2,:])
    plt.plot(time,position[:,2])
    plt.ylim([0,2])
    #plt.ylim([-20,20])
    plt.scatter(lh_traj[:,0],lh_traj[:,2],s=1,c = 'r')
    plt.xlim([80,120])
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')

    plt.figure()

    plt.subplot(311)
    plt.plot(xm[3,:])
    plt.plot(velocity[:,0])

    plt.subplot(312)
    plt.plot(xm[4,:])
    plt.plot(velocity[:,1])

    plt.subplot(313)
    plt.plot(xm[5,:])
    plt.plot(velocity[:,2])


    plt.figure()
    plt.plot(omega) 
    plt.figure()
    #plt.plot(time,velocity[:,0])
    #plt.plot(time,acceleration[:,0])
    #position = truth['Position'].values
    plt.plot(position)
    plt.show()

