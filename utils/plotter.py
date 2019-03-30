import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from pathlib import Path
import numpy as np

def plot_adapted_result(fig_dir, type, result):
    fig_gps = plt.figure(figsize=(16, 16))
    plt.scatter(result['oglng'], result['oglat'], color='black', label='og')
    plt.scatter(result['cv']['lng'], result['cv']['lat'], color='red', label='cv')
    plt.scatter(result['ca']['lng'], result['ca']['lat'], color='orange', label='ca')
    # m = MarkerStyle('octagon')
    plt.scatter(result['adapted']['lng'], result['adapted']['lat'], color='blue', label='adapted')
    plt.xlabel(r'LNG $g$')
    plt.ylabel(r'LAT $g$')
    plt.legend(loc='best', prop={'size': 22})
    plt.grid()
    plt.show()
    plt.savefig(str(fig_dir) + r'\\' + str(type) + 'KF_OG_PRIO_RES.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_result(fig_dir, type,result):
    fig_gps = plt.figure(figsize=(16, 16))
    plt.scatter(result['oglng'], result['oglat'],color='blue',label='og')
    plt.scatter(result[type]['priolng'], result[type]['priolat'],color='orange',label='prio')
    plt.scatter(result[type]['lng'], result[type]['lat'],color='red',label='kf')
    plt.xlabel(r'LNG $g$')
    plt.ylabel(r'LAT $g$')
    plt.legend(loc='best', prop={'size': 22})
    plt.grid()
    plt.show()
    plt.savefig(str(fig_dir) + r'\\'+str(type)+'KF_OG_PRIO_RES.png', dpi=72, transparent=True, bbox_inches='tight')

def plot_llh(fig_dir,type, result):
    end_count = len(result[type]['lng'])
    fig = plt.figure(figsize=(16, 9))
    plt.plot(range(end_count), result[type]['likelihood'], label='likelihood')
    plt.title('Likelihood')
    # plt.show()
    plt.savefig(str(fig_dir) + r'\\'+str(type)+'Kalman-Filter-likelihood.png', dpi=72, transparent=True, bbox_inches='tight')

def plot_epsilons(fig_dir,type, result):
    end_count = len(result[type]['epsilons'])
    fig = plt.figure(figsize=(16, 9))
    plt.plot(range(end_count),result[type]['epsilons'], label='likelihood')
    plt.title('Epsilons')
    # plt.show()
    plt.savefig(str(fig_dir) + r'\\'+str(type)+'Kalman-Filter-epsilons.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_P(fig_dir, matrices):
    fig = plt.figure(figsize=(16, 9))
    Px, Py = [], []

    for p in matrices['P']:
        Px.append(np.diagonal(p)[0])
        Py.append(np.diagonal(p)[2])
    range_of_dts = range(matrices['length'])

    plt.plot(range_of_dts, Px, label='$x$')
    plt.plot(range_of_dts, Py, label='$y$')

    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best', prop={'size': 22})
    plt.xlabel('Filter Step')
    plt.ylabel('Covariance matrix')
    plt.legend(loc='best', prop={'size': 22})
    plt.savefig(str(fig_dir) + r'\Kalman-Filter-CA-XP.png', dpi=72, transparent=True, bbox_inches='tight')

def plot_K(fig_dir, matrices):
    end_count = matrices['length']
    fig = plt.figure(figsize=(16, 9))
    K = matrices['K']
    def get_K_list(index):
        return [i[index,0] for i in K]
    plt.plot(range(end_count), get_K_list(0), label='Kalman Gain for $x$')
    plt.plot(range(end_count), get_K_list(1), label='Kalman Gain for $y$')
    plt.plot(range(end_count), get_K_list(2), label='Kalman Gain for $\dot x$')
    plt.plot(range(end_count), get_K_list(3), label='Kalman Gain for $\dot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
    plt.legend(loc='best', prop={'size': 18})
    plt.savefig(str(fig_dir) + r'\Kalman-Filter-CA-KG.png', dpi=72, transparent=True, bbox_inches='tight')




def plot_m(fig_dir, measurements_count, ma_e, ma_n, acc_down, mp_lng, mp_lat):
    fig_acc = plt.figure(figsize=(16, 9))
    plt.step(range(measurements_count), ma_e, label='$a_x$')
    plt.step(range(measurements_count), ma_n, label='$a_y$')
    plt.step(range(measurements_count), acc_down, label='$a_z$')
    plt.ylabel(r'Acceleration $g$')
    plt.ylim([-2, 2])
    plt.legend(loc='best', prop={'size': 18})

    plt.savefig(str(fig_dir) + r'\Kalman-Filter-CA-Acceleration-Measurements.png', dpi=72, transparent=True,
                bbox_inches='tight')

    fig_gps = plt.figure(figsize=(16, 16))
    plt.scatter(mp_lng, mp_lat)
    plt.xlabel(r'LNG $g$')
    plt.ylabel(r'LAT $g$')
    plt.grid()
    plt.savefig(str(fig_dir) + r'\Kalman-Filter-CA-GPS-Measurements.png', dpi=72, transparent=True, bbox_inches='tight')





def plot_P2(fig_dir, P, end_count):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Covariance Matrix $P$ (after %i Filter Steps)' % (end_count))
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
    plt.savefig(str(fig_dir) + r'\Kalman-Filter-CA-CovarianceMatrix.png', dpi=72, transparent=True, bbox_inches='tight')





def plot_x(fig_dir, end_count):
    fig = plt.figure(figsize=(16, 16))

    plt.subplot(311)
    plt.step(range(end_count), ddxt, label='$\ddot x$')
    plt.step(range(end_count), ddyt, label='$\ddot y$')

    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylabel(r'Acceleration $m/s^2$')
    plt.ylim([-.1, .1])

    plt.subplot(312)
    plt.step(range(end_count), dxt, label='$\dot x$')
    plt.step(range(end_count), dyt, label='$\dot y$')

    plt.ylabel('')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylabel(r'Velocity $m/s$')
    plt.ylim([-1, 1])

    plt.subplot(313)
    plt.step(range(end_count), xt, label='$x$')
    plt.step(range(end_count), yt, label='$y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylabel(r'Position $m$')
    plt.ylim([-1, 1])

    plt.savefig(str(fig_dir) + '\Kalman-Filter-CA-StateEstimated.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_xy(fig_dir):
    fig = plt.figure(figsize=(16, 16))
    plt.plot(xt, yt, label='State', alpha=0.5)
    plt.scatter(xt[0], yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1], yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.savefig(str(fig_dir) + '\Kalman-Filter-CA-Position.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_xyz_acc(acc):
    ma_e = acc['acc_east']
    ma_n = acc['acc_north']
    ma_d = acc['acc_down']
    measurements_count = len(ma_e)
    fig_acc = plt.figure(figsize=(16, 9))
    plt.step(range(measurements_count), ma_e, label='$a_x$')
    plt.step(range(measurements_count), ma_n, label='$a_y$')
    plt.step(range(measurements_count), ma_d, label='$a_z$')
    plt.ylabel(r'Acceleration $g$')
    plt.ylim([-2, 2])
    plt.legend(loc='best', prop={'size': 18})
    plt.show()

    # plt.savefig(str(fig_dir) + r'\Kalman-Filter-CA-Acceleration-Measurements.png', dpi=72, transparent=True, bbox_inches = 'tight')

def plot_ned_acc(fig_dir, time, axis):
    fig_acc = plt.figure(figsize=(16, 9))
    # plt.plot(time, east, color='green')
    # plt.plot(time, north, color='blue')
    plt.plot(time, axis, color='red')
    # plt.xlabel('Acceleration')
    # plt.ylabel('Time')
    # plt.title('ACC NED')
    print('vótmá', (str(fig_dir) + r'\results\figures\Kalman-Filter-CA-Acc_NED.png'))
    #plt.show()
    plt.savefig(r'D:\PyCharmProjects\thesis\data\trolli_playground\constacc\results\figures\Kalman-Filter-CA-Acc_NED.png')#Path(str(fig_dir) + r'\results\figures\Acc.png'))
    print('yes')

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