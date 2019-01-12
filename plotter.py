"""
CC-BY-SA2.0 Lizenz
"""

import matplotlib.pyplot as plt

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
