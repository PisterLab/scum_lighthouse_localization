import numpy as np

class StateEstimator:
    def __init__(self,R_eo):
        self.R_eo = R_eo

class Kalman(StateEstimator):

    def __init__(self,Q,R,xm_0, Pm_0, time, R_eo):

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

    def update(self,measurement,i):
        if len(measurement)== 3 and np.linalg.norm(measurement) < 10:
            np.linalg.norm(measurement)
            S = self.H @ self.Pp[:,:,i] @ self.H.T + self.R
            K = self.Pp[:,:,i] @ self.H.T @ np.linalg.inv(S)    
            print("measurement")
            #calculate measurement's mahalabonis distance from expected measurement distribution
            euc_diff = np.linalg.inv(self.R_eo) @ measurement - self.H @ self.xp[:,i]
            mahal = np.sqrt(euc_diff @ np.linalg.inv(S) @ euc_diff)
            self.mahals.append(mahal)
            print(mahal)

            if mahal < 3:
                self.xm[:,i] = self.xp[:,i] + K @ ( np.linalg.inv(self.R_eo) @ measurement - self.H @ self.xp[:,i]) 
                self.Pm[:,:,i] = (np.eye(6) - K @ self.H) @ self.Pp[:,:,i]
            else:
                self.Pm[:,:,i] = self.Pp[:,:,i]
                self.xm[:,i] = self.xp[:,i]
        
        else:
            self.Pm[:,:,i] = self.Pp[:,:,i]
            self.xm[:,i] = self.xp[:,i]

    def exp_meas(self):
        return self.H @ self.xp

    def meas_available(self,time,lh_traj):

        while(lh_traj[self.lh_index,0] - time < 0):
            self.lh_index += 1

        if lh_traj[self.lh_index,0] - time <= 1/60:
            self.lh_index += 1
            return lh_traj[self.lh_index-1,1:]

        else:
            return [] 

    def predict(self,i,rotations,acceleration):
        earth_accel = np.linalg.inv(self.R_eo) @ np.linalg.inv(rotations[i].to_rotation_matrix()) @ acceleration[i,:]
        #print(earth_accel)
        self.xp[:,i] = self.A@ self.xm[:,i-1] + np.array([0,0,0,earth_accel[0,0],earth_accel[0,1],earth_accel[0,2]])* 1/60 # xm[2,:]
        self.Pp[:,:,i] = self.A @ self.Pm[:,:,i-1] @ self.A.T + self.Q

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

    def __init__(self, R_eo, Q, R, xm_0, Pm_0, time, gnd_to_lh1_pose, gnd_to_lh2_pose):
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
        #print("Pm")
        #print(self.Pm[:,:,i-1])
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
        #print(s_priors)
        #compute prior statistics
        self.xp[:,i] = 1/12 * np.sum(s_priors,axis = 0)
        #print('centering s priors')
        #print(s_priors)
        #print(self.xp[:,i])
        X =  s_priors - self.xp[:,i].reshape((1,6)) 
        #print(X)
        self.Pp[:,:,i] = 1/12 * X.T @ X + self.Q
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
            #print("ztild")
            #print(self.xp[:,i])
            #print(s_priors)
            #print(z_tilds)
            #print(zhat)
            #print('z',azel_meas['z'], 'expected from not sigma points', self.exp_meas(self.xp[:,i],azel_meas['type'],azel_meas['lh_pos'],gnd_to_lh @ self.R_eo) )

            Pzz = 1/12*(np.array(z_tilds) - zhat) @ (np.array(z_tilds) - zhat) + self.R
            #print('Pzz and Pxz')
            #print(Pzz)

            Pxz = 1/12*(s_priors - self.xp[:,i]).T @ (np.array(z_tilds) - zhat)
            Pxz = Pxz.reshape(6,1)
            #print(Pxz)
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
