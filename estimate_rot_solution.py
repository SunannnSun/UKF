import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from quaternion import Quaternion
from ukf import UKF
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an extended kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    # length = min(np.shape(imu['ts'])[1], np.shape(vicon['ts'])[1])
    length = 5000
    # length = 2000

    ukf = UKF()

    vicon_accel = np.zeros((3, length))
    vicon_gyro = np.zeros((3, length))
    vicon_pose = np.zeros((3, length))
    imu_pose = np.zeros((3, length))
    last_q = Quaternion()
    last_ts = vicon['ts'][0,0]-0.0001
    for i in range(length):
        R = vicon['rots'][:,:,i]
        q = Quaternion()
        q.from_rotm(R)
        vicon_pose[:,i] = q.euler_angles()
        vicon_accel[:,i] = ukf.meas_accel(q)
        vicon_gyro[:,i] = ukf.meas_gyro(q, last_q, vicon['ts'][0,i]-last_ts)
        last_q = q
        last_ts = vicon['ts'][0,i]
        # print(str(vicon_pose[:,i]))
        # print(str(ukf.meas_accel(q)) + ", " + str(accel[:,i]))

    last_ts = imu['ts'][0,0]-0.0001
    for i in range(length):
        ts = imu['ts'][0,i]
        ukf.process_update(ts-last_ts)
        meas = np.hstack((accel[:,i], gyro[:,i]))
        ukf.measurement_update(meas)
        imu_pose[:,i] = ukf.state[0].euler_angles()
        last_ts = ts
        print(i,imu_pose[:,i])
        #print "iteration " + str(i)

    #print ukf.state_cov
    #print ukf.meas_accel(ukf.state[0])

    # visualize
    plt.figure(2); plt.clf();
    plt.subplot(131)
    plt.plot(vicon['ts'][0,0:length], vicon_pose[0,:], imu['ts'][0,0:length], imu_pose[0,:])
    plt.subplot(132)
    plt.plot(vicon['ts'][0,0:length], vicon_pose[1,:], imu['ts'][0,0:length], imu_pose[1,:])
    plt.subplot(133)
    plt.plot(vicon['ts'][0,0:length], vicon_pose[2,:], imu['ts'][0,0:length], imu_pose[2,:])
    plt.show()

    plt.figure(1); plt.clf();
    plt.plot(vicon['ts'][0,0:length], vicon_accel[0,:],
             imu['ts'][0,0:length], accel[0,:length])
    plt.figure(2); plt.clf();
    plt.plot(vicon['ts'][0,0:length], vicon_accel[1,:],
             imu['ts'][0,0:length], accel[1,:length])
    plt.figure(3); plt.clf();
    plt.plot(vicon['ts'][0,0:length], vicon_accel[2,:],
             imu['ts'][0,0:length], accel[2,:length])

    return imu_pose[0,:],imu_pose[1,:],imu_pose[2,:]


if __name__ == "__main__":
    estimate_rot(1)
