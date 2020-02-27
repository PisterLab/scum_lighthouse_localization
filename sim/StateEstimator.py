import numpy as np
from py3dmath import Vec3, Rotation
import matplotlib.pyplot as plt

class StateEstimator:
    def __init__(self,R_eo):
        self.R_eo = R_eo
        self.times = []
        self.time_errors = {'negative' : 0, 'oversized' : 0}

    def meas_available(self,time,lh_traj):

        while(lh_traj[self.lh_index,0] - time < 0):
            self.lh_index += 1

        if lh_traj[self.lh_index,0] - time <= 1/60:
            self.lh_index += 1
            return lh_traj[self.lh_index-1,1:]

        else:
            return [] 

    #looks for invalid timestamps 
    def is_valid_time(self,time):
        # check time_delta to see if it is valid
        last_valid_time = self.times[-1]
        #print(time,last_valid_time)
        #print(self.time_errors)
        if time - last_valid_time < 0:
            #invalid, skip
            self.time_errors['negative']+=1

            if self.time_errors['negative'] > 4:
                self.time_errors['negative'] = 0
                self.time_errors['oversized'] = 0
                return True
            else:
                return False

        elif time - last_valid_time > 2:
            self.time_errors['oversized']+=1

            if self.time_errors['oversized'] > 4:
                self.time_errors['negative'] = 0
                self.time_errors['oversized'] = 0
                return True
            else:
                return False

        else: 
            self.time_errors['negative'] = 0
            self.time_errors['oversized'] = 0
            return True

    def meas_available_azel(self,time,lighthouse_df,lh1_gnd, lh2_gnd):
        self.lh_index

        while(lighthouse_df['timestamp_optitrack'].iloc[self.lh_index] - time < 0):
            self.lh_index += 1

        if lighthouse_df['timestamp_optitrack'].iloc[self.lh_index] - time <= 1/60:
            
            if lighthouse_df['azA'].iloc[self.lh_index] > 0:
                #calculate relative
                measurement = {'z': np.deg2rad(lighthouse_df['azA'].iloc[self.lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : lh1_gnd,'lh_num': 1}
                 
            if lighthouse_df['azB'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['azB'].iloc[self.lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : lh2_gnd,'lh_num': 2}

            if lighthouse_df['elA'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['elA'].iloc[self.lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : lh1_gnd,'lh_num': 1}

            if lighthouse_df['elB'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['elB'].iloc[self.lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : lh2_gnd,'lh_num': 2}

            self.lh_index += 1

            return measurement

        else:
            return -1 

    def plot_tracking(self,gnd_df,show = False):
        plt.figure()
        xyz_map = ['X','Y','Z']

        for i in range(1,4):
            plt.subplot(3,1,i)

            plt.scatter(self.times[:],self.xm[i-1,0:len(self.times)],s = 1)
            plt.plot(gnd_df['Time', 'Time'],gnd_df['Position',xyz_map[i-1]],linestyle = '--',color = 'orange')
            #plt.xlim([65,120])
            plt.ylabel(xyz_map[i-1] + ' (m)')
            plt.legend(['Ground Truth','Lighthouse Tracking' ], loc = 'upper right')

            plt.xlabel('Time (s)')

        if show:
            plt.show()        

class Kalman(StateEstimator):

    def __init__(self,Q,R,xm_0, Pm_0, time, R_eo,thresh,t0):
        self.time_errors = {'negative' : 0, 'oversized' : 0}
        self.Q = Q
        self.R = R 
        self.xm = np.zeros((6,len(time)))
        self.xm[:,0] = xm_0

        self.Pm = np.zeros((6,6,len(time)))
        self.Pm[:,:,0] = Pm_0

        self. R_eo = R_eo

        self.xp = np.zeros((6,len(time)))
        self.Pp = np.zeros((6,6,len(time)))
        

        self.A = np.block([[np.eye(3),1/60 * np.eye(3)],
                      [np.zeros((3,3)), np.eye(3)]
                      ])
        
        self.H =     H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]])

        self.mahals = []
        self.lh_index = 0
        self.max_position = 5
        self.thresh = thresh
        self.times = [t0]
        self.i = 1

    def update(self,measurement):
        print(measurement)
        if len(measurement)== 3 and np.linalg.norm(measurement) < 10:
            np.linalg.norm(measurement)
            S = self.H @ self.Pp[:,:,self.i] @ self.H.T + self.R
            K = self.Pp[:,:,self.i] @ self.H.T @ np.linalg.inv(S)    
            #calculate measurement's mahalabonis distance from expected measurement distribution
            euc_diff = np.linalg.inv(self.R_eo) @ measurement - self.H @ self.xp[:,self.i]
            mahal = np.sqrt(euc_diff @ np.linalg.inv(S) @ euc_diff)
            self.mahals.append(mahal)

            if mahal < self.thresh:
                self.xm[:,self.i] = self.xp[:,self.i] + K @ ( np.linalg.inv(self.R_eo) @ measurement - self.H @ self.xp[:,self.i]) 
                print('xm',self.xm[:,self.i])
                self.Pm[:,:,self.i] = (np.eye(6) - K @ self.H) @ self.Pp[:,:,self.i]
            else:
                self.Pm[:,:,self.i] = self.Pp[:,:,self.i]
                self.xm[:,self.i] = self.xp[:,self.i]
        
        else:
            self.Pm[:,:,self.i] = self.Pp[:,:,self.i]
            self.xm[:,self.i] = self.xp[:,self.i]

        self.i += 1
    def exp_meas(self):
        return self.H @ self.xp

    def meas_available(self,time,lh_traj):

        lh_idx = np.searchsorted(lh_traj[:,0],time)
        print(lh_idx,time)

        if lh_idx < len(lh_traj):
            return lh_traj[lh_idx,1:]
        else:
            return []

        #acceleration in body coordinates
    def predict(self,acceleration,omega = None,time= None):
        #rotation is optitrack to body
        #earth_accel =  np.linalg.inv(rotation.to_rotation_matrix()) @ acceleration
        earth_accel = acceleration
        delta_t = time- self.times[-1]
        self.times.append(time)
        self.xp[:,self.i] = self.A@ self.xm[:,self.i-1] + np.array([0,0,0,earth_accel[0],earth_accel[1],earth_accel[2]])* delta_t
        self.Pp[:,:,self.i] = self.A @ self.Pm[:,:,self.i-1] @ self.A.T + self.Q
   
    def get_states(self):
        return self.xm

class EKF(StateEstimator):

    def update(azel_meas):

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

    def exp_meas(xp,meas_type,lh_loc, R):
        print('expected measure diff: ',R @ (xp[0:3] - lh_loc))
        if meas_type == 'az':
            xp_lh = R @ (xp[0:3] - lh_loc)
            angle = np.arctan2(xp_lh[0],xp_lh[2]) 
        else:
            xp_lh = R @ (xp[0:3] - lh_loc)
            angle = np.arctan2(xp_lh[1],xp_lh[2]) 

        return angle 

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

class UKF(StateEstimator):

    def __init__(self, R_eo, Q,

     R, xm_0, Pm_0, time, gnd_to_lh1_pose, gnd_to_lh2_pose):
        self.R_eo = R_eo
        self.Q = Q
        self.R = R 
        self.xm = np.zeros((6,len(time)))
        self.xm[:,0] = xm_0

        self.Pm = np.zeros((6,6,len(time)))
        self.Pm[:,:,0] = Pm_0

        self. R_eo = R_eo

        self.xp = np.zeros((6,len(time)))
        self.Pp = np.zeros((6,6,len(time)))
        

        self.A = np.block([[np.eye(3),1/60 * np.eye(3)],
                      [np.zeros((3,3)), np.eye(3)]
                      ])
        
        self.H =     H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]])

        self.mahals = []
        self.lh_index = 0
        self.gnd_to_lh1_pose = gnd_to_lh1_pose
        self.gnd_to_lh2_pose = gnd_to_lh2_pose
        self.max_position = 5

    def predict(self,i,rotations,acceleration):
        #generate sigma points

        decomp = np.linalg.cholesky(6 * self.Pm[:,:,i-1])
        s_points = []
        for j in range(0,6):
            s_points.append(self.xm[:,i-1] + decomp[:,j])
            s_points.append(self.xm[:,i-1] - decomp[:,j])

        #compute prior sigma points
        earth_accel = np.linalg.inv(self.R_eo) @ np.linalg.inv(rotations[i].to_rotation_matrix()) @ acceleration[i,:]

        s_priors = []
        for j in range(0,12):
            s_priors.append(self.A @ s_points[j] + np.array([0,0,0,earth_accel[0,0],earth_accel[0,1],earth_accel[0,2]])* 1/60)

        s_priors = np.array(s_priors)
        #compute prior statistics
        self.xp[:,i] = 1/12 * np.sum(s_priors,axis = 0)

        X =  s_priors - self.xp[:,i].reshape((1,6)) 
        self.Pp[:,:,i] = 1/12 * X.T @ X + self.Q
        return s_priors

    def exp_meas(self,xp,meas_type,lh_loc, R):
        if meas_type == 'az':
            xp_lh = R @ (xp[0:3] - lh_loc)
            angle = np.arctan2(xp_lh[0],xp_lh[2]) 
        else:
            xp_lh = R @ (xp[0:3] - lh_loc)
            angle = np.arctan2(xp_lh[1],xp_lh[2]) 

        return angle 

    def update(self,azel_meas,i,s_priors):
        if azel_meas != -1:
            if azel_meas['lh_num'] == 1:
                gnd_to_lh = (self.gnd_to_lh1_pose)
            elif azel_meas['lh_num'] == 2:
                gnd_to_lh = (self.gnd_to_lh2_pose)

            #generate predicted measurements from prior s points 
            z_tilds = []
            for point in s_priors:
                z_tilds.append(self.exp_meas(point,azel_meas['type'],azel_meas['lh_pos'],gnd_to_lh @ self.R_eo))
            zhat = np.mean(z_tilds)


            Pzz = 1/12*(np.array(z_tilds) - zhat) @ (np.array(z_tilds) - zhat) + self.R

            Pxz = 1/12*(s_priors - self.xp[:,i]).T @ (np.array(z_tilds) - zhat)
            Pxz = Pxz.reshape(6,1)
            if(i >= 100000):
                assert(False)

            K = Pxz / Pzz 

            
            euc_diff = azel_meas['z'] - zhat
            mahal = np.sqrt(euc_diff * 1/Pzz * euc_diff)
            #mahal = 50
            self.mahals.append(mahal)

            if mahal < 10:

                self.xm[:,i] = self.xp[:,i] + np.squeeze(K) *(azel_meas['z'] - zhat)
                self.Pm[:,:,i] = self.Pp[:,:,i] - Pzz * K @ K.T  


            else:
                self.Pm[:,:,i] = self.Pp[:,:,i]
                self.xm[:,i] = self.xp[:,i]

            self.xm[:,i] = np.clip(self.xm[:,i],-self.max_position,self.max_position)
            #self.Pm[:,:,i] = np.clip(self.Pm[:,:,i],-self.max_position,self.max_position)
        else:
            self.Pm[:,:,i] = self.Pp[:,:,i]
            self.xm[:,i] = self.xp[:,i]

    def meas_available_azel(self,time,lighthouse_df,lh1_gnd, lh2_gnd):
        self.lh_index

        while(lighthouse_df['timestamp_optitrack'].iloc[self.lh_index] - time < 0):
            self.lh_index += 1

        if lighthouse_df['timestamp_optitrack'].iloc[self.lh_index] - time <= 1/60:
            
            if lighthouse_df['azA'].iloc[self.lh_index] > 0:
                #calculate relative
                measurement = {'z': np.deg2rad(lighthouse_df['azA'].iloc[self.lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : lh1_gnd,'lh_num': 1}
                 
            if lighthouse_df['azB'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['azB'].iloc[self.lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : lh2_gnd,'lh_num': 2}

            if lighthouse_df['elA'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['elA'].iloc[self.lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : lh1_gnd,'lh_num': 1}

            if lighthouse_df['elB'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['elB'].iloc[self.lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : lh2_gnd,'lh_num': 2}

            self.lh_index += 1

            return measurement

        else:
            return -1 

class UKF6DoF(StateEstimator):

    def __init__(self, R_eo, Q, R, xm_0, Pm_0, time, gnd_to_lh1_pose, gnd_to_lh2_pose , R_init ,t0, thresh = 5, Rg = .1,lh1_gnd = np.array([1,0,0]), lh2_gnd = np.array([0,0,0]),gravity = True):
        self.numstates = 9
        self.R_eo = R_eo
        self.Q = Q
        assert(len(Q) == self.numstates)

        self.R = R 
        self.xm = np.zeros((9,len(time)))
        self.xm[:,0] = xm_0

        #rotation reference body to earth
        self.Rot_ref = R_init

        self.Pm = np.zeros((9,9,len(time)))
        self.Pm[:,:,0] = Pm_0

        self. R_eo = R_eo

        self.xp = np.zeros((9,len(time)))
        self.Pp = np.zeros((9,9,len(time)))
        

        self.A = np.block([[np.eye(3),1/60 * np.eye(3),np.zeros((3,3))],
                      [np.zeros((3,3)), np.eye(3),np.zeros((3,3))],
                      [np.zeros((3,3)),np.zeros((3,3)),np.eye(3,3)]
                      ])
        
        self.H =     H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]])

        self.mahals = []
        self.lh_index = 0
        self.gnd_to_lh1_pose = gnd_to_lh1_pose
        self.gnd_to_lh2_pose = gnd_to_lh2_pose
        self.lh1_gnd = lh1_gnd
        self.lh2_gnd = lh2_gnd
        self.max_position = 5
        self.eulers = []
        self.rotations = []
        self.i = 1
        self.times = [t0]
        self.size = 0;
        self.thresh = thresh
        self.Rg = np.eye(3) * Rg
        self.time_errors = {'negative' : 0, 'oversized' : 0}
        self.grav_vector = gravity

    def get_eulers(self):
        return np.array(self.eulers)

    def dynamics_step(self,state,acceleration,delta_t,omega):

        att_err = Rotation(0,0,0,0).from_rotation_vector(state[6:9])

        #acceleration is rotated by reference rotation and attitude error
        if self.grav_vector:
            earth_accel =  np.squeeze(att_err.to_rotation_matrix() @ self.Rot_ref @ (acceleration) )- np.array([0,0,9.8])
        else:
            earth_accel =  np.squeeze(att_err.to_rotation_matrix() @ self.Rot_ref @ (acceleration))
        #print(self.times[self.i],earth_accel)
        
        pos_curr = state[0:3]
        vel_curr = state[3:6]
        err_curr = state[6:9]

        pos_next = pos_curr + vel_curr * delta_t
        vel_next = (vel_curr + earth_accel * delta_t).reshape((3,1))

        #gyro stuff
        ref_obj = Rotation(1,0,0,0).from_rotation_matrix(self.Rot_ref) #body to earth rot
        rot_cross = ref_obj.to_rotation_vector().to_cross_product_matrix().T
        err_next = err_curr.reshape((3,1)) + ((np.eye(3) - 1/2*np.nan_to_num(rot_cross)) @ omega * delta_t ).reshape((3,1))


        next_state = np.array([pos_next[0], pos_next[1], pos_next[2], vel_next[0], 
            vel_next[1], vel_next[2], err_next[0],err_next[1], err_next[2]])

        return next_state
    

    def predict(self,acceleration,omega = None,time = None):

        #generate sigma points
        self.times.append(time)
        numstates = 9

        decomp = np.linalg.cholesky(numstates * self.Pm[:,:,self.i-1])
        s_points = []
        for j in range(0,numstates):
            s_points.append(self.xm[:,self.i-1] + decomp[:,j])
            s_points.append(self.xm[:,self.i-1] - decomp[:,j])

        #compute prior sigma points
        s_priors = []
        for j in range(0,2*numstates):
            s_priors.append(self.dynamics_step(s_points[j],acceleration,1/60,omega))

        s_priors = np.array(s_priors)

        #compute prior statistics
        self.xp[:,self.i] = 1/(2*numstates) * np.sum(s_priors,axis = 0)

                 

        #generate centered s_priors for Pp calculation
        X =  s_priors - self.xp[:,self.i].reshape((1,numstates)) 

        self.Pp[:,:,self.i] = 1/(2*numstates) * X.T @ X + self.Q

        #reset attitude error
        att_err = Rotation(0,0,0,0).from_rotation_vector(self.xp[6:9,self.i])
        #update reference
        self.Rot_ref = att_err.to_rotation_matrix() @ self.Rot_ref
        #generate covariance rotation matrix
        half_rot = Rotation(0,0,0,0).from_rotation_vector(1/2*self.xp[6:9,self.i]).to_rotation_matrix()
        tran_mat = np.block([[np.eye(6), np.zeros((6,3))],
                             [np.zeros((3,6)), half_rot]])   
        #reset attitude error       
        self.xp[6:9,self.i] = half_rot @ self.xp[6:9,self.i] - self.xp[6:9,self.i]   
        #rotate covariance
        self.Pp[:,:,self.i] = tran_mat @ self.Pp[:,:,self.i] @ tran_mat.T                             
        #print("attitude error after: ", self.xp[6:9,i])
        #print("covariance after: ", self.Pp[6:9,6:9,i])

        return s_priors

    def exp_meas(self,xp,meas_type,lh_loc, R):
        #print('expected measure diff: ',R @ (xp[0:3] - lh_loc))
        if meas_type == 'az':
            xp_lh = R @ (xp[0:3] - lh_loc)
            angle = np.arctan2(xp_lh[0],xp_lh[2]) 
        else:
            xp_lh = R @ (xp[0:3] - lh_loc)
            angle = np.arctan2(xp_lh[1],xp_lh[2]) 

        return angle 

    def update(self,azel_meas,s_priors,acceleration):
        if azel_meas == None:
            return 

        if azel_meas != -1:
            #print("measurement")
            if azel_meas['lh_num'] == 1:
                gnd_to_lh = (self.gnd_to_lh1_pose)
            elif azel_meas['lh_num'] == 2:
                gnd_to_lh = (self.gnd_to_lh2_pose)

            #generate predicted measurements from prior s points 
            z_tilds = []
            for point in s_priors:
                z_tilds.append(self.exp_meas(point,azel_meas['type'],azel_meas['lh_pos'],gnd_to_lh @ self.R_eo))
            zhat = np.mean(z_tilds)


            Pzz = 1/(2*self.numstates)*(np.array(z_tilds) - zhat) @ (np.array(z_tilds) - zhat) + self.R
            Pxz = 1/(2*self.numstates)*(s_priors - self.xp[:,self.i]).T @ (np.array(z_tilds) - zhat)
            Pxz = Pxz.reshape(self.numstates,1)

            if(self.i >= 100000):
                assert(False)

            K = Pxz / Pzz 

            
            euc_diff = azel_meas['z'] - zhat
            mahal = np.sqrt(euc_diff * 1/Pzz * euc_diff)
            #mahal = 50
            self.mahals.append(mahal)

            if mahal < self.thresh:

                self.xm[:,self.i] = self.xp[:,self.i] + np.squeeze(K) *(azel_meas['z'] - zhat)
                self.Pm[:,:,self.i] = self.Pp[:,:,self.i] - Pzz * K @ K.T  


                #reset attitude error
                att_err = Rotation(0,0,0,0).from_rotation_vector(self.xm[6:9,self.i])
                #update reference
                self.Rot_ref = att_err.to_rotation_matrix() @ self.Rot_ref
                #generate covariance rotation matrix
                half_rot = Rotation(0,0,0,0).from_rotation_vector(1/2*self.xm[6:9,self.i]).to_rotation_matrix()
                tran_mat = np.block([[np.eye(6), np.zeros((6,3))],
                                     [np.zeros((3,6)), half_rot]])   
                #reset attitude error       
                self.xm[6:9,self.i] = half_rot @ self.xm[6:9,self.i] - self.xm[6:9,self.i]   
                #rotate covariance
                self.Pm[:,:,self.i] = tran_mat @ self.Pm[:,:,self.i] @ tran_mat.T


            else:
                print(euc_diff)
                print(Pzz)
                print(Pxz)
                self.Pm[:,:,self.i] = self.Pp[:,:,self.i]
                self.xm[:,self.i] = self.xp[:,self.i]

            self.xm[:,self.i] = np.clip(self.xm[:,self.i],-self.max_position,self.max_position)
            #self.Pm[:,:,i] = np.clip(self.Pm[:,:,i],-self.max_position,self.max_position)
        else:
            '''
            K, z, zhat, Pzz = self.gravity_update(acceleration,s_priors)

            #self.xm[:,self.i] = self.xp[:,self.i] + np.squeeze(K) @(z - zhat)
            #self.Pm[:,:,self.i] = self.Pp[:,:,self.i] - K @ Pzz @ K.T  
            #print(self.Pm[:,:,self.i])
            print(np.squeeze(K) @(z - zhat))
            print(z-zhat)
            #reset attitude error
            att_err = Rotation(0,0,0,0).from_rotation_vector(self.xm[6:9,self.i])
            #update reference
            self.Rot_ref = att_err.to_rotation_matrix() @ self.Rot_ref
            #generate covariance rotation matrix
            half_rot = Rotation(0,0,0,0).from_rotation_vector(1/2*self.xm[6:9,self.i]).to_rotation_matrix()
            tran_mat = np.block([[np.eye(6), np.zeros((6,3))],
                                 [np.zeros((3,6)), half_rot]])   
            #reset attitude error       
            self.xm[6:9,self.i] = half_rot @ self.xm[6:9,self.i] - self.xm[6:9,self.i]   
            #rotate covariance
            self.Pm[:,:,self.i] = tran_mat @ self.Pm[:,:,self.i] @ tran_mat.T    
            '''
            
            self.Pm[:,:,self.i] = self.Pp[:,:,self.i]
            self.xm[:,self.i] = self.xp[:,self.i]
        euler_pose = Rotation(0,0,0,0).from_rotation_matrix(self.Rot_ref).to_euler_YPR()
        self.eulers.append(euler_pose)
        self.rotations.append(self.Rot_ref)
        self.i += 1
        self.size += 1

    def gravity_update(self,acceleration,s_priors):
        #generate predicted measurements from prior s points 
        z_tilds = []
        for point in s_priors:
            
            #gravity direction is rotated by reference rotation and attitude error
            att_err = Rotation(0,0,0,0).from_rotation_vector(point[6:9])
            grav_vector =  np.linalg.inv(att_err.to_rotation_matrix() @ self.Rot_ref )@  np.array([0,0,1])

            #generate expected gravity rotation from acceleration
            z_tilds.append(grav_vector)
        #print(np.squeeze(np.array(z_tilds)))
        z_tilds = np.squeeze(np.array(z_tilds))
        zhat = np.mean(z_tilds,0)
        #normalize
        zhat = zhat / np.linalg.norm(zhat)
        #print(z_tilds-zhat)
        #TODO: make Rg adaptive based on amount of acceleration deviating from 1 g
        adaptive_factor = 10 ** (1+2*((np.linalg.norm(acceleration) - 9.8)**2))
        Pzz = 1/(2*self.numstates)*(z_tilds - zhat).T @ (z_tilds - zhat) + self.Rg 
        Pxz = 1/(2*self.numstates)*(s_priors - self.xp[:,self.i]).T @ (z_tilds - zhat)
        Pxz = Pxz.reshape(self.numstates,3)

        if(self.i >= 100000):
            assert(False)

        K = Pxz @ np.linalg.inv( Pzz) 
        z = acceleration / np.linalg.norm(acceleration)

        return K, z, zhat, Pzz

    def meas_available_azel(self,time,lighthouse_df):


        self.lh_index
        #print(lighthouse_df['timestamp_optitrack'].iloc[self.lh_index], time)
        if 'timestamp_optitrack' not in lighthouse_df.columns:
            column = 'timestamp'
        else:
            column = 'timestamp_optitrack'

        if self.lh_index >= len(lighthouse_df[column]):
            return -1
        
        while(lighthouse_df[column].iloc[self.lh_index] - time < 0):
            #print(lighthouse_df[column].iloc[self.lh_index], time)
            self.lh_index += 1

            if self.lh_index >= len(lighthouse_df[column]):
                #we've run out of lighthouse measurements, so return. 

                return -1

        if lighthouse_df[column].iloc[self.lh_index] - time <= 1/60:
            
            if lighthouse_df['azA'].iloc[self.lh_index] > 0:
                #calculate relative
                measurement = {'z': np.deg2rad(lighthouse_df['azA'].iloc[self.lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : self.lh1_gnd,'lh_num': 1}
                 
            if lighthouse_df['azB'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['azB'].iloc[self.lh_index]) - np.pi/2,'type': 'az', 'lh_pos' : self.lh2_gnd,'lh_num': 2}

            if lighthouse_df['elA'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['elA'].iloc[self.lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : self.lh1_gnd,'lh_num': 1}

            if lighthouse_df['elB'].iloc[self.lh_index] > 0:
                measurement = {'z': np.deg2rad(lighthouse_df['elB'].iloc[self.lh_index]) - np.pi/2,'type': 'el', 'lh_pos' : self.lh2_gnd,'lh_num': 2}

            self.lh_index += 1

            return measurement

        else:
            return -1 
    def plot_states(self):
        plt.figure()
        plt.subplot(321)
        plt.plot(self.times,self.xm[0,0:len(self.times)])
        plt.title('X')
        plt.ylabel(' Position (m)')

        plt.subplot(323)
        plt.plot(self.times,self.xm[1,0:len(self.times)])
        plt.title('Y')
        plt.ylabel(' Position (m)')

        plt.subplot(325)
        plt.plot(self.times,self.xm[2,0:len(self.times)])
        plt.title('Z')
        plt.xlabel('Time (s)')
        plt.ylabel(' Position (m)')

        plt.subplot(322)
        plt.plot(self.times,self.xm[3,0:len(self.times)])
        plt.title('Vx')
        plt.ylabel(' Velocity (m/s)')

        plt.subplot(324)
        plt.plot(self.times,self.xm[4,0:len(self.times)])
        plt.title('Vy')
        plt.ylabel(' Velocity (m/s)')

        plt.subplot(326)
        plt.plot(self.times,self.xm[5,0:len(self.times)])
        plt.title('Vz')
        plt.ylabel(' Velocity (m/s)')
        plt.xlabel('time (s)')

        plt.show()

