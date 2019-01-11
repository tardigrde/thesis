"""
CC-BY-SA2.0 Lizenz
"""
import matplotlib.pyplot as plt
from kalman import Kalman
import nmea_parser
import imu_data_parser
import numpy as np
import initial_params


def get_acceleration_data(path_imu):
    imu_list_of_dicts = imu_data_parser.get_imu_dictionary(path_imu)
    return imu_list_of_dicts


def get_gps_data(path_gps):
    gps_list_of_dicts = nmea_parser.get_gps_dictionary(path_gps)
    return gps_list_of_dicts


def pass_gps_list(gps):
    time, ln, la, v, t, hdop = [], [], [], [], [], []
    timestamps = sorted(list(gps.keys()))
    for t in timestamps:
        values = gps[t]
        v.append(values['v'])
        hdop.append(values['hdop'])
        time.append(values['time'])
        ln.append(values['lng'])
        la.append(values['lat'])
    return {'ln': ln, 'la': la, 'v': v, 't': t, 'hdop': hdop, 'time': time, }


def pass_acc_list(acc):
    timestamps = sorted(list(acc.keys()))
    acc_time, acc_east, acc_north, acc_down = [], [], [], []
    for t in timestamps:
        values = acc[t]
        acc_east.append(values['acc_east'])
        acc_north.append(values['acc_north'])
        acc_down.append(values['acc_down'])
        acc_time.append(values['time'])
    return {'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down, 'acc_time': acc_time}


def _pass_std_devs(acc):
    acc_lists = pass_acc_list(acc)
    std_dev_acc_east = np.std(acc_lists['acc_east'])
    std_dev_acc_north = np.std(acc_lists['acc_north'])
    return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}


def interpolate_and_trim_gps_data(acc, gps):
    gps_lists = pass_gps_list(gps)
    acc_lists = pass_acc_list(acc)

    acc_time = acc_lists['acc_time']
    acc_east = acc_lists['acc_east']
    acc_north = acc_lists['acc_north']
    acc_down = acc_lists['acc_down']


    gps_time = gps_lists['time']
    gps_lng = gps_lists['ln']
    gps_lat = gps_lists['la']
    gps_hdop = gps_lists['hdop']
    gps_v = gps_lists['v']

    interpolated_lng = np.interp(acc_time, gps_time, gps_lng).tolist()
    interpolated_lat = np.interp(acc_time, gps_time, gps_lat).tolist()
    interpolated_hdop = np.interp(acc_time, gps_time, gps_hdop).tolist()
    interpolated_v = np.interp(acc_time, gps_time, gps_v).toList()
    # interpolated_vlng = np.interp(acc_time, gps_time, gps_vln).tolist()
    # interpolated_vlat = np.interp(acc_time, gps_time, gps_vla).tolist()
    # interpolated_t = np.interp(acc_time, gps_time, gps_t).toList()

    """
    TODO:
        - refactor this messssssssssss
    """

    if (len(acc_time) == len(acc_down) == len(acc_east) == len(acc_north) == len(interpolated_lat) == len(
            interpolated_lng) == len(interpolated_v) == len(interpolated_hdop)):
        count = len(acc_time)
        print('LEN DOES EQUALS;\nit is: ', count)
        trim_count = count / 20
        print('LEN/20 IS: ', trim_count)

        del (acc_time[:trim_count])
        del (acc_east[:trim_count])
        del (acc_north[:trim_count])
        del (acc_down[:trim_count])
        del (interpolated_lat[:trim_count])
        del (interpolated_lng[:trim_count])
        del (interpolated_hdop[:trim_count])
        del (interpolated_v[:trim_count])

        print('TRIMMED COUNT: ', len(acc_east))

        return {
            'acc_time': acc_time, 'lng': interpolated_lng, 'lat': interpolated_lat,
            'v': interpolated_v, 'hdop': interpolated_hdop,
            'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down
        }
    else:
        print('NEM')





def get_kalmaned_coordinates(path_acc, path_gps):
    acc = get_acceleration_data(path_acc)
    gps = get_gps_data(path_gps)

    # http: // geopandas.org /

    init_params = initial_params.get_initial_params()
    P = init_params['P']
    H = init_params['H']
    R = init_params['R']
    Q = init_params['Q']
    A = init_params['F']
    I = init_params['I']

    std_devs = _pass_std_devs(acc)

    interpolated_attribute_table = interpolate_and_trim_gps_data(acc, gps)
    table = interpolated_attribute_table()

    acc_time = interpolated_attribute_table['acc_time']
    acc_east = interpolated_attribute_table['acc_east']
    acc_north = interpolated_attribute_table['acc_north']
    acc_down = interpolated_attribute_table['acc_down']
    gps_lng = interpolated_attribute_table['lng']
    gps_lat = interpolated_attribute_table['lat']
    gps_v = interpolated_attribute_table['v']
    gps_hdop = interpolated_attribute_table['hdop']

    # fused = {**acc, **gps}
    # sorted_timestamps_of_all_measurements = sorted(list(fused.keys()))
    # print('Length of steps is: {}'.format(len(sorted_timestamps_of_all_measurements)))

    m = len(acc_time)  # Measurements

    sp = 1.0  # Sigma for position !!!!! THIS IS HDOP
    px = 0.0  # x Position
    py = 0.0  # y Position

    mpx = np.array(px + sp * np.random.randn(m))
    mpy = np.array(py + sp * np.random.randn(m))

    # Generate GPS Trigger
    GPS = np.ndarray(m, dtype='bool')
    GPS[0] = True
    # Less new position updates
    for i in range(1, m):
        if i % 10 == 0:
            GPS[i] = True
        else:
            mpx[i] = mpx[i - 1]
            mpy[i] = mpy[i - 1]
            GPS[i] = False

    # Acceleration --> MOVE THIS TO THE FOR LOOP
    sa = 0.1  # Sigma for acceleration THIS IS STD_DEV((acc_x+acc_y)/2)
    ax = 0.0  # in X
    ay = 0.0  # in Y

    # GPS --> MOVE THIS TO FOOR LOOP
    mx = np.array(ax + sa * np.random.randn(m))
    my = np.array(ay + sa * np.random.randn(m))

    measurements = np.vstack((mpx, mpy, mx, my))
    print(measurements.shape)
    return

    def plot_m():
        fig = plt.figure(figsize=(16, 9))
        plt.subplot(211)
        plt.step(range(m), mpx, label='$x$')
        plt.step(range(m), mpy, label='$y$')
        plt.ylabel(r'Position $m$')
        plt.title('Measurements')
        plt.ylim([-10, 10])
        plt.legend(loc='best', prop={'size': 18})

        plt.subplot(212)
        plt.step(range(m), mx, label='$a_x$')
        plt.step(range(m), my, label='$a_y$')
        plt.ylabel(r'Acceleration $m/s^2$')
        plt.ylim([-1, 1])
        plt.legend(loc='best', prop={'size': 18})

        plt.savefig('Kalman-Filter-CA-Measurements.png', dpi=72, transparent=True, bbox_inches='tight')

    plot_m()

    # Preallocation for Plotting
    xt = []
    yt = []
    dxt = []
    dyt = []
    ddxt = []
    ddyt = []
    Zx = []
    Zy = []
    Px = []
    Py = []
    Pdx = []
    Pdy = []
    Pddx = []
    Pddy = []
    Kx = []
    Ky = []
    Kdx = []
    Kdy = []
    Kddx = []
    Kddy = []

    def savestates(x, Z, P, K):
        xt.append(float(x[0]))
        yt.append(float(x[1]))
        dxt.append(float(x[2]))
        dyt.append(float(x[3]))
        ddxt.append(float(x[4]))
        ddyt.append(float(x[5]))
        Zx.append(float(Z[0]))
        Zy.append(float(Z[1]))
        Px.append(float(P[0, 0]))
        Py.append(float(P[1, 1]))
        Pdx.append(float(P[2, 2]))
        Pdy.append(float(P[3, 3]))
        Pddx.append(float(P[4, 4]))
        Pddy.append(float(P[5, 5]))
        Kx.append(float(K[0, 0]))
        Ky.append(float(K[1, 0]))
        Kdx.append(float(K[2, 0]))
        Kdy.append(float(K[3, 0]))
        Kddx.append(float(K[4, 0]))
        Kddy.append(float(K[5, 0]))

        for filterstep in range(m):

            # Time Update (Prediction)
            # ========================
            # Project the state ahead
            x = A * x

            # Project the error covariance ahead
            P = A * P * A.T + Q

            # Measurement Update (Correction)
            # ===============================
            # if there is a GPS Measurement
            if GPS[filterstep]:
                # Compute the Kalman Gain
                S = H * P * H.T + R
                K = (P * H.T) * np.linalg.pinv(S)

                # Update the estimate via z
                Z = measurements[:, filterstep].reshape(H.shape[0], 1)
                y = Z - (H * x)  # Innovation or Residual
                x = x + (K * y)

                # Update the error covariance
                P = (I - (K * H)) * P

            # Save states for Plotting
            savestates(x, Z, P, K)

    # plot_P()
    #
    # plot_P2()
    #
    # plot_K()
    #
    # plot_x()
    #
    # plot_xy()

    dist = np.cumsum(np.sqrt(np.diff(xt) ** 2 + np.diff(yt) ** 2))
    print('Your drifted %dm from origin.' % dist[-1])



def do_pothole_extraction(acc, gps):
    """

    :param acc: Dict of lists of acceleration data.
    :param gps: Dict of lists of gps data.
    :return:
    """
    interpolated_attribute_table = interpolate_and_trim_gps_data(acc, gps)
    acc_time = interpolated_attribute_table['acc_time']
    acc_east = interpolated_attribute_table['acc_east']
    acc_north = interpolated_attribute_table['acc_north']
    acc_down = interpolated_attribute_table['acc_down']
    gps_lng = interpolated_attribute_table['lng']
    gps_lat = interpolated_attribute_table['lat']
    gps_v = interpolated_attribute_table['v']
    gps_hdop = interpolated_attribute_table['hdop']
    print('Count of acc_down is {}'.format(len(acc_down)))
    print('Biggest value is {}'.format(max(acc_down)))
    print('Lowest value is {}'.format(min(acc_down)))

    out_weka = './teszt/szeged_trolli_teszt/nointerpolation/for_weka.csv'
    with open(out_weka, 'w') as out:
        for d in acc_down:
            out.write(str(d) + ',' + '' + '\n')
    return

    p_lng = []
    p_lat = []
    # for lng, lat, down in zip(gps_lng, gps_lat, acc_down):
    #     if down > 1.2 or 0.4 < down < 0.8:
    #         p_lng.append(lng)
    #         p_lat.append(lat)
    # pothole = [value for value in acc_down if value > 1.2 or 0.4 < value < 0.8]
    # print('Count of pothole is {}'.format(len(pothole)))
    # plt.plot(acc_down, 'b--')
    # plt.show()
    fig = plt.figure()
    plt.plot(gps_lng, gps_lat, 'b--', p_lng, p_lat, 'rs')
    plt.show()
    fig.savefig('first_potholes_manually_extracted.pdf', dpi=fig.dpi)
    return acc_down

"""
--------------------------------------------------------------------------------------------
"""
"""
    TODO:
    - check/adjust velocity
    - try 2x1D
    - make it OO for real
    - make cross-terms 0
    - try like this: https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1
    - try it like it was before interpolation kinda ok, but not really
    - geohash 


    https://github.com/akshaychawla/1D-Kalman-Filter
    https://dsp.stackexchange.com/questions/8860/kalman-filter-for-position-and-velocity-introducing-speed-estimates
    https://dsp.stackexchange.com/questions/38045/even-more-on-kalman-filter-for-position-and-velocity?noredirect=1&lq=1
    https://dsp.stackexchange.com/questions/48343/python-how-can-i-improve-my-1d-kalman-filter-estimate
    https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python
    https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
    https://www.reddit.com/r/computervision/comments/35y4kj/looking_for_a_python_example_of_a_simple_2d/
    https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1 !!!!!!!!!!!!!!!!
    https://balzer82.github.io/Kalman/
    https://towardsdatascience.com/kalman-filter-an-algorithm-for-making-sense-from-the-insights-of-various-sensors-fused-together-ddf67597f35e
    https://gist.github.com/manicai/922976
"""


def plot_P():
    fig = plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(range(len(measurements[0])), Px, label='$x$')
    plt.plot(range(len(measurements[0])), Py, label='$y$')
    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best', prop={'size': 22})
    plt.subplot(212)
    plt.plot(range(len(measurements[0])), Pddx, label='$\ddot x$')
    plt.plot(range(len(measurements[0])), Pddy, label='$\ddot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best', prop={'size': 22})


def plot_P2():
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Covariance Matrix $P$ (after %i Filter Steps)' % (m))
    ylocs, ylabels = plt.yticks()
    # set the locations of the yticks
    plt.yticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    xlocs, xlabels = plt.xticks()
    # set the locations of the yticks
    plt.xticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    plt.xlim([-0.5, 5.5])
    plt.ylim([5.5, -0.5])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig('Kalman-Filter-CA-CovarianceMatrix.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_K():
    fig = plt.figure(figsize=(16, 9))
    plt.plot(range(len(measurements[0])), Kx, label='Kalman Gain for $x$')
    plt.plot(range(len(measurements[0])), Ky, label='Kalman Gain for $y$')
    plt.plot(range(len(measurements[0])), Kdx, label='Kalman Gain for $\dot x$')
    plt.plot(range(len(measurements[0])), Kdy, label='Kalman Gain for $\dot y$')
    plt.plot(range(len(measurements[0])), Kddx, label='Kalman Gain for $\ddot x$')
    plt.plot(range(len(measurements[0])), Kddy, label='Kalman Gain for $\ddot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
    plt.legend(loc='best', prop={'size': 18})


def plot_x():
    fig = plt.figure(figsize=(16, 16))

    plt.subplot(311)
    plt.step(range(len(measurements[0])), ddxt, label='$\ddot x$')
    plt.step(range(len(measurements[0])), ddyt, label='$\ddot y$')

    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylabel(r'Acceleration $m/s^2$')
    plt.ylim([-.1, .1])

    plt.subplot(312)
    plt.step(range(len(measurements[0])), dxt, label='$\dot x$')
    plt.step(range(len(measurements[0])), dyt, label='$\dot y$')

    plt.ylabel('')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylabel(r'Velocity $m/s$')
    plt.ylim([-1, 1])

    plt.subplot(313)
    plt.step(range(len(measurements[0])), xt, label='$x$')
    plt.step(range(len(measurements[0])), yt, label='$y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylabel(r'Position $m$')
    plt.ylim([-1, 1])

    plt.savefig('Kalman-Filter-CA-StateEstimated.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_xy():
    fig = plt.figure(figsize=(16, 16))
    plt.plot(xt, yt, label='State', alpha=0.5)
    plt.scatter(xt[0], yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1], yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.savefig('Kalman-Filter-CA-Position.png', dpi=72, transparent=True, bbox_inches='tight')
