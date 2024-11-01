import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]

options['DRIVE_IN_CIRCLE'] = False
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = False
options['RECIEVE_INPUTS'] = False

# This filter uses four states x, y, x dot, y dot
# Works ok ish for turns
# However, if the vehicle makes a circle it doesn't converge at all.
# To overcome, this we need to use a five state transition matrix.
class KalmanFilter:
    def __init__(self):
        # Initial State
        self.x = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[1000., 0., 0., 0.],
                            [0., 1000., 0., 0.],
                            [0., 0., 1000., 0.],
                            [0., 0., 0., 1000.]])

        # Next State Function
        self.F = np.matrix([[1.0, 0., 0., 0.],
                            [0., 1.0, 0., 0.],
                            [0., 0., 1.0, 0.],
                            [0., 0., 0., 1.0]])

        # Measurement Function
        self.H = np.matrix([[1.0, 0., 0., 0.],
                            [0., 1.0, 0., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[0.01, 0.0],
                            [0.0, 0.01]])

        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
        
    def predict(self, dt):
        self.F[0,2] = dt
        self.F[1,3] = dt
        self.x = self.F * self.x
        self.P =  self.F * self.P * np.transpose(self.F)
        return
    
    def measure_and_update(self,measurements, dt):
        self.F[0,2] = dt
        self.F[1,3] = dt
        Z = np.matrix(measurements)
        y = np.transpose(Z) - self.H * self.x
        S = self.H * self.P * np.transpose(self.H) + self.R
        K = self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (self.I - K * self.H) * self.P
        # In each iteration the uncertainity values should be slightly incremented to make sure
        # The Kalman filter doesn't become too confident on its current state.
        #self.P[0,0] += 0.001 # Commented. Felt like not required for position value
        #self.P[1,1] += 0.001 # Commented. Felt like not required for position value
        self.P[2,2] += 0.001 # Incrementing the velocity uncertainity alone because this will ultimately affect the position
        self.P[3,3] += 0.001
        return [self.x[0], self.x[1]]

    def recieve_inputs(self, u_steer, u_pedal):
        return

sim_run(options,KalmanFilter)
