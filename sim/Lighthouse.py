import numpy as np
import cv2 as opencv
import pandas as pd
import matplotlib.pyplot as plt
class Lighthouse:
    def __init__(self,x=0,y=0,z=0,pose=np.eye(3)):

        self.pose = pose
                 

        self.translation = np.array([[1,0,0,-x],
                                 [0,1,0,-y],
                                 [0,0,1,-z]])
        self.K = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])

        self.P = np.matmul(np.matmul(self.K,self.pose),self.translation)

        self.cvRot = None
        self.cvTrans = None
        self.cvCam = None


    def setOpenCVParams(self, tvec, rvec, K):
        self.cvRvec = rvec
        self.cvTvec = tvec
        self.cvK = K


        rot, jacobian = opencv.Rodrigues(rvec)
        self.pose = np.squeeze(np.array(rot)) 

        modified_t =  -np.linalg.inv(self.pose) @ np.squeeze(tvec)

        self.translation = np.array([[1,0,0,-modified_t[0]], [0,1,0,-modified_t[1]], [0,0,1,-modified_t[2]]]) 

        self.K = np.squeeze(np.array(K))
        self.P = self.pose @ self.translation

#this class generates xyz coordinates from lighthouse azimuths
class Triangulator:  
    def __init__(self,lighthouse1, lighthouse2):
        self.P1 = lighthouse1.P
        self.P2 = lighthouse2.P
        
        self.xcam1 = None
        self.ycam1 = None

    def triangulate(self,az1, el1, az2, el2):
        #calculate triangulation (remember a on lighthouse is actually b
        #here

        xcam1 = np.tan(az1 - np.pi/2) #focal distance of 1 (virtual)
        ycam1 = np.tan((el1 - np.pi/2))
        
        self.xcam1 = xcam1
        self.ycam1 = ycam1

        xcam2 = np.tan(az2 - np.pi/2); #focal distance of 1 (virtual)
        ycam2 = np.tan((el2 - np.pi/2))
        
        #stack A matrix (entry for each lighthouse is X x (PX) = 0) direct linear transform
        A = [xcam1 * self.P1[2,:] - self.P1[0,:],
             ycam1 * self.P1[2,:] - self.P1[1,:],
             xcam2 * self.P2[2,:] - self.P2[0,:],
             ycam2 * self.P2[2,:] - self.P2[1,:]]
        
        #find SVD 
        u,s,vt = np.linalg.svd(A)
        v= np.transpose(vt)
        #solution is final column of V
        xhat = v[:,3]/v[3,3]
        return (np.squeeze(xhat)[0],np.squeeze(xhat)[1],np.squeeze(xhat)[2])    

#this function will align the lighthouse data to the vicon data. It will
#return lighthouse data timestamp column that is temporally aligned with respect to the opititrack data
def align_data(lh_data,take_name,optitrack_data = None):

    #time offsets (lh - optitrack)
    offsets = {'take1' : 284.6-85.4039,
                'take2' : 257.359 - 2.94922,
                'take3' : 58.08 - 5.671,
                'take4' : 150.123 - 96.4074,
                'take5' : 116.671 - 84.6218,
                'take3/20200226-004043' : 33.3 - 25.1}

    aligned_timestamps = lh_data['timestamp_seconds'] - offsets[take_name] 
    return aligned_timestamps

#loads lighthouse data. Will optionally align data to optitrack data; these offsets must be manually performed  
def load_lh(lighthouse_filename,take_name, align = False):

    #load lighthouse data file
    with open(lighthouse_filename) as file:
        if take_name == 'take1' or take_name == 'take5':
            lighthouse_df = pd.read_csv(file, header=None, names = ['azA', 'elA', 'azB', 'elB', 'timestamp_10.82 Hz'], dtype = float)
        else: 
            lighthouse_df = pd.read_csv(file, header=None,lineterminator = '\n', sep = ',', names = ['azA', 'elA', 'azB', 'elB', 'timestamp_10.82 Hz'],dtype = float)

        lighthouse_df['timestamp_seconds'] = lighthouse_df['timestamp_10.82 Hz'] / 10.82e6
        
        

    if align:
        lighthouse_df['timestamp_optitrack'] = align_data(lighthouse_df,take_name)
    else: 
        lighthouse_df['timestamp_optitrack'] = lighthouse_df['timestamp_seconds']

    return lighthouse_df

#lh_data is the raw lighthouse dataframe
def triangulate_scum(lh_data,lighthouse1, lighthouse2):

    #create triangulator
    triangulator = Triangulator(lighthouse1,lighthouse2)

    #we triangulate every time we get a new measurement, which means we need to use the most recent 
    #for the calculation values of the other angles
    
    azA = [np.nan]
    azB = [np.nan]
    elA = [np.nan]
    elB = [np.nan]

    #print(lh_data)
    xcam1 = [np.nan]
    ycam1 = [np.nan] 
    xcam2 = [np.nan]
    ycam2 = [np.nan]

    traj = []
    cam_points = []
    #iterate through every timestep
    for i in range(0,len(lh_data)):

        if abs(lh_data['azA'].iloc[i]) < 180:
            azA.append(lh_data['azA'].iloc[i] * np.pi/180.0)
            xcam1.append( np.tan(azA[-1] - np.pi/2) )#focal distance of 1 (virtual)
    
        if abs(lh_data['azB'].iloc[i]) < 180:
            azB.append(lh_data['azB'].iloc[i] * np.pi/180.0)
            xcam2.append( np.tan(azB[-1]- np.pi/2) )#focal distance of 1 (virtual)

        if abs(lh_data['elA'].iloc[i]) < 180:
            elA.append(lh_data['elA'].iloc[i] * np.pi/180.0)
            ycam1.append(np.tan((elA[-1] - np.pi/2)))

        if abs(lh_data['elB'].iloc[i]) < 180:
            elB.append(lh_data['elB'].iloc[i] * np.pi/180.0)
            ycam2.append( np.tan((elB[-1] - np.pi/2)))

        if not np.isnan([azA[-1],elA[-1],azB[-1],elB[-1]]).any():
            point = triangulator.triangulate(az1 = azA[-1], el1 = elA[-1], az2 = azB[-1], el2 = elB[-1])
            traj.append([lh_data['timestamp_optitrack'].iloc[i], point[0],point[1],point[2]])
            cam_points.append([lh_data['timestamp_optitrack'].iloc[i], xcam1[-1],ycam1[-1],xcam2[-1],ycam2[-1]])

    return np.squeeze(np.array(traj)), np.squeeze(np.array(cam_points))

#generates cam projection points for opencv lighthouse calibration
def generate_cam_points(lh_data):

    xcam1 = [np.nan]
    ycam1 = [np.nan] 
    xcam2 = [np.nan]
    ycam2 = [np.nan]

    traj = []
    cam_points = []
    #iterate through every timestep
    for i in range(0,len(lh_data)):

        if abs(lh_data['azA'].iloc[i]) < 180 and  lh_data['azA'].iloc[i] > 0:
            azA = lh_data['azA'].iloc[i] * np.pi/180.0
            xcam1.append(np.tan(azA - np.pi/2) )#focal distance of 1 (virtual)
            if(xcam1[-1] < -5):
                print(azA, xcam1[-1])
    
        if abs(lh_data['azB'].iloc[i]) < 180:
            azB = lh_data['azB'].iloc[i] * np.pi/180.0
            xcam2.append(np.tan(azB- np.pi/2) )#focal distance of 1 (virtual)

        if abs(lh_data['elA'].iloc[i]) < 180:
            elA = lh_data['elA'].iloc[i] * np.pi/180.0
            ycam1.append(np.tan(elA - np.pi/2))

        if abs(lh_data['elB'].iloc[i]) < 180:
            elB = lh_data['elB'].iloc[i] * np.pi/180.0
            ycam2.append(np.tan(elB - np.pi/2))

        if not np.isnan([xcam1[-1],ycam1[-1],xcam2[-1],ycam2[-1]]).any():
            cam_points.append([lh_data['timestamp_optitrack'].iloc[i], xcam1[-1],ycam1[-1],xcam2[-1],ycam2[-1]])

    return np.squeeze(np.array(cam_points))

#take in azimuth and elevation in degrees 
#this solves the equation X_gnd * P = Y_lh, solving for the P that minimizes error
def calibrate(scum_gnd_df,initial_cam1, initial_cam2,lighthouse_cams):


    cal_window_dict = {'take5': [2000, 3500]}
    begin = 1000
    end = 1500
    
    '''
    begin = 100
    end = 5000
    '''
    #loop through each point 
    
    scum_gnd_df.set_index(scum_gnd_df['Time','Time'].values, inplace = True)
    #print(scum_gnd_df['Time','Time'])
    #print(scum_gnd_df.index)
    Pcam = initial_cam1.copy()
    
    gnd_cam = []
    lh_cam_1 = [ list([row[1],row[2]]) for row in lighthouse_cams[100:400] ]
    lh_cam_1 = []
    plt.figure()
    plt.plot(lighthouse_cams[:,1:5])
    plt.title('Lighthouse Camera Points')
    #print(lh_cam_1)
    for i in range(begin,end):
        current_point_time = lighthouse_cams[i,0]
        gnd_index = scum_gnd_df.index.get_loc(current_point_time,method = 'nearest')
        gnd_row = scum_gnd_df.iloc[gnd_index]
        gnd_x = np.interp(current_point_time, scum_gnd_df['Time','Time'],scum_gnd_df['Position','X']) 
        gnd_y = np.interp(current_point_time, scum_gnd_df['Time','Time'],scum_gnd_df['Position','Y']) 
        gnd_z = np.interp(current_point_time, scum_gnd_df['Time','Time'],scum_gnd_df['Position','Z']) 
        
        gnd_point = list(gnd_row['Position'].values[0:3].astype('float32'))
        if (not np.isnan(gnd_point).any()) and (not (abs(lighthouse_cams[i,1:3]) > 5).any())  and (not (abs(lighthouse_cams[i,0]) > 900)) and (not (lighthouse_cams[i,0] < 0)):
            lh_cam_1.append([lighthouse_cams[i][1], lighthouse_cams[i][2]])
            gnd_cam.append(gnd_point)

    mat = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,0]])

    #print(np.array([gnd_cam]))
    #print(np.array([lh_cam_1]).astype('float32'))
    K = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,1] ],dtype='float32')
    #initial guesses
    rvec_guess = np.array([[ -0.15],
                           [ -3.08 ],
                           [0.18]], dtype='float32')
    tvec_guess = np.array([[ 0.285139  ],
                           [-0.81256074],
                           [ 1.6812717 ]], dtype='float32')
    felipe_mat_5 = np.array([[ 0.99584145, -0.07031334, 0.05792963],
                        [-0.08038641, -0.97739281,0.19555386],
                        [ 0.04286996, -0.1993974,-0.97898051]])

    m_to_b = np.array([[ 0.88273753, -0.30231239,0.35969664],
                        [ 0.30012588,0.95179021,0.06340231],
                        [-0.36152305,0.05198667,0.93091271]], dtype = 'float32')



    print('Felipe matrix: ')
    print(np.linalg.inv(felipe_mat_5)   @ m_to_b)

    retval, rvec, tvec = opencv.solvePnP(np.array([gnd_cam]),np.array([lh_cam_1]).astype('float32'),K,None,rvec = rvec_guess, tvec = tvec_guess, useExtrinsicGuess = 1)
    retval, rvec, tvec, inliers = opencv.solvePnPRansac(np.array([gnd_cam]),np.array([lh_cam_1]).astype('float32'),K,None,rvec = rvec_guess, tvec = tvec_guess, useExtrinsicGuess = 1)

    print("pnp")
    print('retval: ',retval)
    print('rvec: ',rvec)
    print('tvec:', tvec)

    print("lh1 rotation mat")
    rot, jacobian = opencv.Rodrigues(rvec)
    print(rot)
    #print(np.linalg.inv(rot) @ tvec)
    print(tvec)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(gnd_cam)
    plt.title('Ground Cam')
    plt.subplot(2,1,2)
    plt.plot(lh_cam_1)
    plt.title('Lighthouse Cam 1')

    lh1_obj = Lighthouse()
    lh1_obj.setOpenCVParams(tvec = tvec,rvec = rvec,K = K)

    imagePoints, jacob = opencv.projectPoints(np.array([gnd_cam]),lh1_obj.cvRvec,lh1_obj.cvTvec,lh1_obj.cvK,None)



    

    cvproj1 = np.squeeze(np.array(imagePoints))

    rot, rando = opencv.Rodrigues(rvec)
    xyz_camera = rot @ (np.array(gnd_cam).T + np.linalg.inv(rot) @ lh1_obj.cvTvec)

    plt.figure()
    plt.subplot(2,1,1)
    #plt.plot(xyz_camera[0,:]/xyz_camera[2,:])
    plt.plot(np.array(lh_cam_1)[:,0])
    plt.title("Lighthouse 1 projecttion check")
    plt.plot(cvproj1[:,0])
    plt.legend(['Lighthouse','OpenCV Projectect Truth'])

    plt.subplot(2,1,2)
    plt.plot(np.array(lh_cam_1)[:,1])
    plt.plot(cvproj1[:,1])
    plt.legend(['Lighthouse','OpenCV Projectect Truth'])



###################################################################
    gnd_cam = []
    lh_cam_2 = []
    
    #print(lh_cam_1)
    for i in range(begin,end):
        current_point_time = lighthouse_cams[i,0]
        gnd_index = scum_gnd_df.index.get_loc(current_point_time,method = 'nearest')
        gnd_row = scum_gnd_df.iloc[gnd_index]
        gnd_x = np.interp(current_point_time, scum_gnd_df['Time','Time'],scum_gnd_df['Position','X']) 
        gnd_y = np.interp(current_point_time, scum_gnd_df['Time','Time'],scum_gnd_df['Position','Y']) 
        gnd_z = np.interp(current_point_time, scum_gnd_df['Time','Time'],scum_gnd_df['Position','Z']) 
        
        gnd_point = list(gnd_row['Position'].values[0:3].astype('float32'))
        if (not np.isnan(gnd_point).any()) and (not (abs(lighthouse_cams[i,3:5]) > 5).any()) and (not (abs(lighthouse_cams[i,0]) > 900)) and (not (lighthouse_cams[i,0] < 0)):
            lh_cam_2.append([lighthouse_cams[i][3], lighthouse_cams[i][4]])
            gnd_cam.append(gnd_point)

    mat = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,0]])

    #print(np.array([gnd_cam]))
    #print(np.array([lh_cam_1]).astype('float32'))
    K = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,1] ],dtype='float32')

    #initial guesses, should improve convergence
    rvec_guess = np.array([[ -0.12],
                            [ 3.31 ],
                            [ -0.09554877]], dtype='float32')

    tvec_guess = np.array([[ 1.1421727 ],
                            [-0.90],
                            [ 1.5230292 ]], dtype='float32')

    felipe_mat_5 = np.array([[ 0.99511576,0.02454935,-0.09561352],
                            [ 0.0507425,-0.95803291,0.28213142],
                            [-0.08467475,-0.28560509,-0.95459935]])

    print('Felipe matrix inverted: ')
    print(np.linalg.inv(felipe_mat_5))

    retval, rvec, tvec = opencv.solvePnP(np.array([gnd_cam]),np.array([lh_cam_2]).astype('float32'),K,None,rvec = rvec_guess, tvec = tvec_guess, useExtrinsicGuess = 1)
    retval, rvec, tvec, inliers= opencv.solvePnPRansac(np.array([gnd_cam]),np.array([lh_cam_2]).astype('float32'),K,None,rvec = rvec_guess, tvec = tvec_guess, useExtrinsicGuess = 1)
    
    print("pnp lh 2")
    print('retvat: ',retval)
    print('rvec: ',rvec)
    print('tvec:' ,tvec)

    print("lh2 rotation mat")
    rot, jacobian = opencv.Rodrigues(rvec)
    print(rot)
    print(np.linalg.inv(rot) @ tvec)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(gnd_cam)
    plt.subplot(2,1,2)
    plt.plot(lh_cam_2)


    lh2_obj = Lighthouse()
    lh2_obj.setOpenCVParams(tvec = tvec,rvec = rvec,K = K)

    imagePoints, jacob = opencv.projectPoints(np.array([gnd_cam]),lh2_obj.cvRvec,lh2_obj.cvTvec,lh2_obj.cvK,None)



    cvproj2 = np.squeeze(np.array(imagePoints))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.array(lh_cam_2)[:,0])
    plt.title("Lighthouse 2 projection check")
    plt.plot(cvproj2[:,0])
    plt.legend(['Lighthouse','OpenCV Projectect Truth'])

    plt.subplot(2,1,2)
    plt.plot(np.array(lh_cam_2)[:,1])
    plt.plot(cvproj2[:,1])
    plt.legend(['Lighthouse','OpenCV Projectect Truth'])



    return lh1_obj, lh2_obj, cvproj1, cvproj2

def plot_raw_data(azimuth1_gnd, azimuth2_gnd, elevation1_gnd, elevation2_gnd, lighthouse_df, scum_gnd_df, show = True):
    #################################
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(-azimuth1_gnd*180/3.14159 +90)
    plt.ylim([50,150])
    plt.title('Az 1')

    plt.subplot(2,2,3)
    #print(lighthouse_df['azA'])

    plt.plot(lighthouse_df[ lighthouse_df['azA'] > -360 ]['azA'])
    plt.ylim([50,150])
    plt.title('Az A')

    plt.subplot(2,2,2)
    plt.plot(-azimuth2_gnd*180/3.14159 +90)
    plt.ylim([50,150])
    plt.title('Az 2')

    plt.subplot(2,2,4)
    #print(lighthouse_df['azA'])
    plt.plot(lighthouse_df[ lighthouse_df['azB'] > -360 ]['azB'])
    plt.ylim([50,150])
    plt.title('Az B')

#############
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(scum_gnd_df['Time','Time'],-elevation1_gnd*180/3.14159 +90)
    plt.ylim([50,150])
    plt.title('el 1')

    plt.subplot(2,2,3)
    #print(lighthouse_df['azA'])
    plt.plot(lighthouse_df[ lighthouse_df['elA'] > -360 ]['timestamp_optitrack'], lighthouse_df[ lighthouse_df['elA'] > -360 ]['elA'])
    plt.ylim([50,150])
    plt.title('el A')

    plt.subplot(2,2,2)
    plt.plot(scum_gnd_df['Time','Time'],-elevation2_gnd*180/3.14159 +90)
    plt.ylim([50,150])
    plt.title('el 2')

    plt.subplot(2,2,4)
    #print(lighthouse_df['azA'])
    plt.plot(lighthouse_df[ lighthouse_df['elB'] > -360 ]['timestamp_optitrack'], lighthouse_df[ lighthouse_df['elB'] > -360 ]['elB'])
    plt.ylim([50,150])
    plt.title('el b')

    plt.figure()
    plt.plot(lighthouse_df[ lighthouse_df['elA'] > -360 ]['timestamp_optitrack'], lighthouse_df[ lighthouse_df['elA'] > -360 ]['elA'])
    plt.plot(scum_gnd_df['Time','Time'],-elevation1_gnd*180/3.14159 +90)
    plt.title('elevation comparison')
    plt.ylim([50,150])

    if show:
        plt.show()
