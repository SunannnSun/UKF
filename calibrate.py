from cProfile import label
import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt


if __name__ == "__main__":
    g = 9.81

    # LOAD IMU
    data_num = 3
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]


    # LOAD VICON DATASET AND PROCESS ROTATION MATRIX
    rot = vicon['rots']
    ts_vicon = vicon['ts'][0, :]
    rot_list = [rot[:, :, i] for i in range(rot.shape[-1])]
    
    quat_list = [Quaternion() for i in range(rot.shape[-1])]
    quat_list = [quat_list[index].from_rotm(entry) for index, entry in enumerate(rot_list)]

    euler_list = [entry.euler_angles() for entry in quat_list]
    euler_arr  = np.vstack(euler_list)


    # PLOT ROLL/PITCH/YAW FROM VICON
    colors = ['red', 'blue', 'lime']
    legends = ['roll', 'pitch', 'yaw']

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for k in range(3):
        axs[k].plot(np.arange(len(euler_list)), euler_arr[:, k], color=colors[k], label=legends[k]+ "_vicon")
        axs[k].legend()
    axs[0].set_title("Vicon Roll/Pitch/Yaw")


    # CALIBRATE ACCELERATOR MEASURE
    """
    - Calibrate betas first so roll pitch align with vicon from the beginning
    - Calibrate alpha_z so the magnitude remains at +g in the beginning
    - Calibrate the rest of alphas so the roll pitch align with vicon later on
    """

    accel = accel.astype(int)

    alpha_accel = [30, 40, 35]
    beta_accel  = [510, 501, 500]


    accel_proc = np.zeros((3, T))
    for k in range(3):
        accel_proc[k, :] = (accel[k, :] - beta_accel[k]) * 3300 / (1023 * alpha_accel[k])

    accel_proc[0, :] = -accel_proc[0, :]
    accel_proc[1, :] = -accel_proc[1, :]


    # COMPUTE ROLL PITCH FROM ACCEL ASSUMING STATIONARY
    """
    Rotation Matrix to Euler Angle: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    Derivation: https://mwrona.com/posts/accel-roll-pitch/

    Note: Although IMU experiences gravitational acceleration acting downwards, the reading is positive. Hence, the 
    accelerometer vector at stationary should be [0, 0, +g] as opposed to -g in the derivation post, and the negative
    sign in computing pitch
    """

    roll_accel = np.arctan(accel_proc[1, :]/accel_proc[2, :])
    pitch_accel = np.arctan(-accel_proc[0, :]/np.linalg.norm(accel_proc[0:3, :], axis=0))

    axs[0].set_title('Accelerometer Calibration: IMU Roll/Pitch')
    axs[0].plot(range(T), roll_accel, '--', color=colors[0], label=legends[0]+'_accel')
    axs[0].legend()
    axs[1].plot(range(T), pitch_accel, '--', color=colors[1], label=legends[1]+'_accel')
    axs[1].legend()


    colors = ['red', 'blue', 'lime']
    legends = ['a_x', 'a_y', 'a_z']
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for k in range(3):
        axs[k].plot(np.arange(accel_proc.shape[1]), accel_proc[k, :], color=colors[k], label=legends[k])
        if k == 2:
            axs[k].plot(np.arange(accel_proc.shape[1]), 9.81*np.ones((accel_proc.shape[1],)), 'k', label = "g")
        axs[k].legend()
    
    axs[0].set_title('Accelerometer Calibration: Acceleration Magnitude')

    
    """

    accel_mag = np.linalg.norm(accel_proc, axis=0)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    ax.plot(np.arange(T), accel_mag)
    ax.set_title("IMU Acceleration Magnitude")

    # RETRIEVE GROUND TRUTH ANGULAR VELOCITY
    w_xyz = np.zeros((len(quat_list)-1, 3))
    dt = np.zeros((len(quat_list)-1, ))
    for i in np.arange(1, len(quat_list)):
        dt[i-1] = ts_vicon[i] - ts_vicon[i-1]

        q_next = R.from_matrix(rot_list[i])
        q_prev = R.from_matrix(rot_list[i-1])
        dq = q_next * q_prev.inv()
        w_xyz[i-1, :] = dq.as_rotvec()

        # q_next = quat_list[i]
        # q_prev = quat_list[i-1]
        # dq = q_next.__mul__(q_prev.inv())
        # w_xyz[i-1, :] = dq.axis_angle()/dt
    dt = np.mean(dt)


    colors = ['red', 'blue', 'lime']
    legends = ['w_x', 'w_y', 'w_z']
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for k in range(3):
        axs[k].plot(np.arange(w_xyz.shape[0]), w_xyz[:, k]/dt, color=colors[k], label=legends[k])
        axs[k].legend()
    axs[0].set_title('Vicon Angular Velocity')

    
    
    # PLOT GYROSCOPE MEASURE
    gyro = gyro.astype(int)
    alpha_gyro = [250, 250, 250]
    beta_gyro  = [369, 373, 376]

    gyro_proc = np.zeros((3, T))
    for k in range(3):
        gyro_proc[k, :] = (gyro[k, :] - beta_gyro[k]) * 3300 / (1023 * alpha_gyro[k])


    colors = ['red', 'blue', 'lime']
    legends = ['w_x', 'w_y', 'w_z']
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for k in range(3):
        axs[k].plot(np.arange(gyro_proc.shape[1]), gyro_proc[k, :], color=colors[k], label=legends[k])
        axs[k].legend()
    axs[0].set_title('IMU Angular Velocity')



    

    """


    plt.show()

    pass