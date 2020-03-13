import numpy as np


def get_initial_params():


    # timestep
    dt = 0.01
    
    # initial state matrix(4x1)
    X = np.asarray([0, 0, 0, 0])
    
    # initial state covariance(4)
    P = np.eye(4)
    
    # F - state transition matrix (4x4)
    F = np.eye(4)
    F[0, 1], F[2, 3] = dt, dt
    
    # B - control matrix(4x2)
    B = np.zeros((4, 2))
    B[0, 0], B[1, 1] = (dt ** 2) / 2, (dt ** 2) / 2
    B[2, 0], B[3, 1] = dt, dt
    
    # H -measurment matrix(4)
    H = [[1, 0, 0, 0], [0, 1, 0, 0]]
    
    # R - measurment covariance matrix(4x4)
    R = np.zeros((4, 4))
    
    # Q - process measurment matrix(4x4)
    Q = np.zeros((4, 4))
    I = np.eye(4)

    return {'X': X, 'P': P, 'F': F, 'I': I, 'H': H, 'R': R, 'Q': Q}
