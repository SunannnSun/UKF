from quaternion import Quaternion
import numpy as np
from scipy import linalg
import math

class UKF:
    def __init__(self):
        #IMU scales and biases
        self.accel_cal = np.array([[-1.,0,0],[0,-1.,0],[0,0,1.]])
        self.accel_cal *= 103/9.81
        self.accel_bias = np.array([511, 501, 503])
        self.gyro_cal = np.array([[0,0,1.],[1,0,0],[0,1,0]])
        self.gyro_cal *= 60
        self.gyro_bias = np.array([369.5, 371.5, 377])

        #Initialize system state and state covariance
        self.state = [Quaternion(), np.array([0.,0,0])]
        self.state_cov = np.diag([1.,1,1,1,1,1])

        self.Wp = np.zeros((6,12))

        #UKF Parameters
        #Process Covariances
        self.proc_noise = np.diag([0.01,0.01,0.01,0.1,0.1,0.1])
        self.meas_noise = (np.diag([3.,3,3,3,3,3]))**2

    #Measurement functions
    def calibrate_accel(self, accel):
        accel_calib = np.matmul(self.accel_cal, accel)
        accel_calib += self.accel_bias
        return accel_calib

    def calibrate_gyro(self, gyro):
        gyro_calib = np.matmul(self.gyro_cal, gyro)
        gyro_calib += self.gyro_bias
        return gyro_calib

    #Directly try to find orientation from the accelerometer
    def meas_accel(self, q):
        accel_q = q.inv()*Quaternion(0, [0, 0, 9.81])*q
        return self.calibrate_accel(accel_q.vec())

    def meas_gyro(self, q, last_q, dt):
        dq = q*last_q.inv()
        #convert to angular velocity vector
        theta = 2*math.acos(dq.scalar())
        vec = dq.vec()
        vec = vec/np.linalg.norm(vec)
        if (dt < 0.005):
            dt = 0.005
        return self.calibrate_gyro(vec*theta/dt)

    def unscented_trans_state(self, sigma):
        #perform unscented transform on the state
        #compute error vectors iteratively
        errors = np.zeros((3, np.shape(sigma)[1]))
        error_sum = np.ones(3)
        mean = self.state
        ii = 0
        while np.linalg.norm(error_sum) > 0.01:
            index = 0
            for s in sigma.T:
                q = Quaternion(s[0], s[1:4])
                e = q*mean[0].inv()
                errors[:, index] = e.axis_angle()
                index += 1
            error_sum = np.mean(errors, 1)
            error_q = Quaternion()
            error_q.from_axis_angle(error_sum)
            mean[0] = error_q*mean[0]
            ii += 1
        #angular velocity portion is easy
        mean[1] = np.mean(sigma, 1)[4:7]

        #covariance now is fairly straightforward
        self.Wp = np.vstack((errors[:,:], sigma[4:7,:]-np.expand_dims(mean[1],1)))
        cov = np.cov(self.Wp)*11/12

        return (mean, cov)

    def unscented_trans_meas(self, sigma):
        mean = np.mean(sigma, 1)

        cov = np.cov(sigma-np.expand_dims(mean,1))*11/12
        cross_cov = np.cov(np.vstack((self.Wp, sigma-np.expand_dims(mean,1))))[0:6,6:]*11/12

        return (mean, cov, cross_cov)

    def process_update(self, dt):
        #angular velocity stays the same
        #Update orientation, assuming angular velocity
        self.state_cov += self.proc_noise*(dt)
        sigma = self.gen_sigma()
        new_sigma = np.zeros((7, 12))
        #propagate sigma points
        index = 0
        for s in sigma:
            dq = Quaternion()
            dq.from_axis_angle(s[1]*dt)
            q = s[0]
            new_sigma[:,index] = np.hstack(((dq*q).q, s[1]))
            index += 1

        (self.state, self.state_cov) = self.unscented_trans_state(new_sigma)

    def measurement_update(self, Z):
        sigma = self.gen_sigma()
        new_sigma = np.zeros((6,12))
        #propagate sigma points
        index = 0
        for s in sigma:
            q = s[0]
            new_sigma[0:3, index] = self.meas_accel(q)
            new_sigma[3:6, index] = self.calibrate_gyro(s[1])
            index += 1
        #compute mean, cov
        (meas_mean, meas_cov, meas_cross_cov) = self.unscented_trans_meas(new_sigma)
        innovation = Z - meas_mean
        meas_cov += self.meas_noise

        kalman_gain = np.matmul(meas_cross_cov, np.linalg.inv(meas_cov))
        state_raw = np.matmul(kalman_gain, np.expand_dims(innovation,1)).T
        q = Quaternion()
        q.from_axis_angle(state_raw[0][0:3])
        self.state[0] = q*self.state[0]
        self.state[1] = state_raw[0][3:6] + self.state[1]
        self.state_cov -= np.matmul(np.matmul(kalman_gain, meas_cov), kalman_gain.T)

    def gen_sigma(self):
        n = np.shape(self.state_cov)[1]
        S = linalg.sqrtm(self.state_cov)
        #Find 0 centered sigma pts
        W = np.hstack((S*math.sqrt(n), -S*math.sqrt(n)))
        X = []
        for w in W.T:
           q_w = Quaternion()
           q_w.from_axis_angle(w[0:3])
           q = self.state[0]*q_w
           X.append([q, w[3:6]+self.state[1]])
        return X



#ukf = UKF()
#print "------------------"
#ukf.process_update(1)
#print ukf.state[0]
#print ukf.state[1]
#print ukf.state_cov
#ukf.measurement_update([511., 501, 605, 370, 370, 370])
#print ukf.state[0].axis_angle()
#print ukf.state[1]
#print ukf.state_cov
#
#print ukf.meas_accel(ukf.state[0])
#print ukf.calibrate_gyro(ukf.state[1])
