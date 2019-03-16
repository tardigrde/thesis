import numpy as np


def get_initial_params():
    dt = 0.1

    I = np.eye(4)
    print(I, I.shape)

    # initial state matrix(4x1)
    X = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    print('X:\n', X, X.shape)

    # initial state covariance(4)
    P = np.diag([100.0, 100.0, 10.0, 10.0, 2.0, 2.0])
    print('P:\n', P, P.shape)

    # F - state transition matrix (4x4)
    F = np.matrix([[1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2, 0.0],
                   [0.0, 1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2],
                   [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    print('F:\n', F, F.shape)

    # B - control matrix(4x2)
    B = np.zeros((4, 2))
    B[0, 0], B[1, 1] = (dt ** 2) / 2, (dt ** 2) / 2
    B[2, 0], B[3, 1] = dt, dt

    # H - measurment matrix(4)
    H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    print('H:\n', H, H.shape)

    # R - measurment covariance matrix(4x4)
    R = np.matrix([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    print('R:\n', R, R.shape)

    # Q - process noise covariance matrix(6x6)
    G = np.matrix([[0.5 * dt ** 2],
                   [0.5 * dt ** 2],
                   [dt],
                   [dt],
                   [1.0],
                   [1.0]])
    Q = G * G.T
    print('G:\n', Q, Q.shape)

    return {'X': X, 'P': P, 'F': F, 'I': I, 'H': H, 'R': R, 'Q': Q}
