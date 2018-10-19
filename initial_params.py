import numpy as np

class Initial_params:
    def __init__(self):
        
        # timestep
        dt = 0.01

        # initial state matrix(4x1)
        self.X = np.asarray([0, 0, 0, 0])
        
        # initial state covariance(4)
        self.P = np.eye(4)
    
        # F - state transition matrix (4x4)
        self.F = np.eye(4)
        self.F[0,1], self.F[2,3] = dt, dt    

        # B - control matrix(4x2)
        self.B = np.zeros((4,2))
        self.B[0,0], self.B[1,1] = (dt**2)/2,(dt**2)/2
        self.B[2,0], self.B[3,1] = dt, dt

        # H -measurment matrix(4)
        self.H = [[1,0,0,0],[0,1,0,0]]

        # R - measurment covariance matrix(4x4)
        self.R = np.zeros((4,4))
        
        # Q - process measurment matrix(4x4)
        self.Q = np.zeros((4,4))
    
    def get_initial_parameters(self):
        return {'X': self.X, 'P': self.P, 'F': self.F, 'B': self.B, 'H': self.H, 'R': self.R, 'Q': self.Q}