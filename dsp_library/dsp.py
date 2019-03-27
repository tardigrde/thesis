from __future__ import division
from scipy.signal import find_peaks, butter
import matplotlib.pyplot as plt
from utils import fuser
from utils import plotter
import pandas as pd
import numpy as np
import peakutils
import copy
import numpy as np
from scipy import pi
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def do_fft(window):
    """
    https://ericstrong.org/fast-fourier-transforms-in-python/
    https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
    Args:
        window:

    Returns:

    """
    if not len(window) == 100:
        print('_________NOOOOOOOOOOOOOOOOOOOOOOOOOOOPE_____')
        return

    # Sampling rate and time vector
    start_time = 0  # seconds
    end_time = 1  # seconds
    sampling_rate = 100  # Hz
    N = (end_time - start_time) * sampling_rate  # array size

    # Vibration data generation
    time = np.linspace(start_time, end_time, N)
    vib_data = window


    plt.plot(time[0:100], vib_data[0:100])
    plt.xlabel('Time')
    plt.ylabel('Vibration (g)')
    plt.title('Time Domain (Healthy Machinery)')
    plt.show()

    # Nyquist Sampling Criteria
    T = 1 / sampling_rate  # inverse of the sampling rate
    x = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    # FFT algorithm
    yr = fft(vib_data)  # "raw" FFT with both + and - frequencies
    y = 2 / N * np.abs(yr[0:np.int(N / 2)])  # positive freqs only


    # Plotting the results
    plt.plot(x, y)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Vibration (g)')
    plt.title('Frequency Domain (Healthy Machinery)')
    plt.show()

def do_HPF(acc_down_subset):
    """
    https://plot.ly/python/fft-filters/
    Args:
        acc_down_subset:

    Returns:

    """

    fc = 20  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 5  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute a low-pass filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2))
    w = np.blackman(N)
    h = h * w
    h = h / np.sum(h)

    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[(N - 1) // 2] += 1

    filtered = np.convolve(acc_down_subset, h)
    # TODO:
    # - interpolate
    # - make HPF work
    if not len(filtered) == len(acc_down_subset) and len(filtered) > len(acc_down_subset):
        print('OOOOOOOOOOOPS', len(filtered), len(acc_down_subset))
        diff = len(filtered)-len(acc_down_subset)
        print(diff)
        filtered_windows_cut = np.delete(filtered, [i for i in range(diff)])

    plt.plot(acc_down_subset, color='red')
    plt.plot(filtered_windows_cut, color='green')
    plt.title("Hamming windows filtered")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()

    return filtered_windows_cut



def find_max_peaks(z_axis, height):
    peaks, _ = find_peaks(z_axis, height)
    return peaks, _


def do_hamming(dataset, n):
    window = np.hamming(n)
    if len(dataset) != n: window = np.hamming(len(dataset))
    windowed_data_og = np.multiply(dataset, window)
    return windowed_data_og


def do_splitting(axis, hamming=True, window_size=100):
    n = window_size
    m = round(n / 4) # overlap_size
    if hamming == False:
        windows = [axis[i:i + n] for i in range(0, len(axis), n - m)]
    else:
        windows = [do_HPF(axis[i:i + n], n) for i in range(0, len(axis), n - m)]
        #windows = [do_hamming(axis[i:i + n], n) for i in range(0, len(axis), n - m)]
    return windows


def segment_data(acc, gps):
    # https://docs.scipy.org/doc/numpy-1.14.5/reference/generated/numpy.fft.fft.html#numpy.fft.fft
    import time
    start_time = time.time()

    dataset = fuser.interpolate_and_trim_data(acc, gps)
    windows = {
        "east": do_splitting(dataset['east']),
        "north": do_splitting(dataset['north']),
        "down": do_splitting(dataset['down']),
        "time": do_splitting(dataset['time']),
        "lng": do_splitting(dataset['lng'], False),
        "lat": do_splitting(dataset['lat'], False),
    }
    if not len(windows['time']) == len(windows['east']) == len(windows['north']) == len(windows['down']) ==len(windows['lng']) == len(windows['lat']):
        print('ERROR: the length of the windows is not the same!')
        return
    print("--- Segmentation took %s seconds ---" % (time.time() - start_time))
    return windows


def classify_windows(acc, gps, dir_path):
    windows = segment_data(acc, gps)

    down_subsets_filtered = [do_HPF(down_subset) for down_subset in windows['down']]
    # st = calculate_stats(windows)
    # plotter.plot_ned_acc(fig_dir, t, down_acc)
    #fftd_windows = [do_fft(down_subset) for down_subset in windows['down']]












def choose_potholes(stats):
    # TODO:
    # - high pass filter
    # - fft
    down_means = []

    bad_segments = {
        "lng": [],
        "lat": [],
        "class": [],
    }
    counter = 0
    for st in stats:
        for ln, lt in zip(st['lng'], st['lat']):
            bad_segments['lng'].append(ln)
            bad_segments['lat'].append(lt)
            if st['max']['d'] > 0.2 or st['min']['d'] < -0.2:
                bad_segments['class'].append('1')
                counter = counter + 1
            else:
                bad_segments['class'].append('0')
        # print(st['max']['d'],st['min']['d'])

    print('len of windows: {}, or: {}'.format(len(stats), len(bad_segments['lng'])))
    print('len of potholes: {}'.format(counter))
    print(bad_segments)
    df = pd.DataFrame(bad_segments)
    fuser.convert_result_to_shp(df, r'D:\PyCharmProjects\thesis\data\20190115\harmadik\results\potholes\result')
    print('done')


def calculate_stats(windows):
    stats_array = []
    stats = {
        "lng": [],
        "lat": [],
        "max": {},
        "min": {},
        "max_peaks": {},
        "min_peaks": {},
        "mean": {},
        "std_dev": {},
        "variance": {},
        "p2p": {},
        "sma": {},
        "ar_coef": {},
        "tilt_angles": {},
        "rms": {},
        "abs_correlation": "",
    }
    lengths = []
    # max_indexes = peakutils.indexes(down, thres=1, thres_abs=True)
    for e, n, d, ln, lt in zip(windows['east'], windows['north'], windows['down'], windows['lng'], windows['lat']):
        stats['mean']['e'] = np.mean(e)
        stats['mean']['n'] = np.mean(n)
        stats['mean']['d'] = np.mean(d)
        stats['min']['e'] = min(e)
        stats['min']['n'] = min(n)
        stats['min']['d'] = min(d)
        stats['max']['e'] = max(e)
        stats['max']['n'] = max(n)
        stats['max']['d'] = max(d)
        stats['max_peaks']['e'] = len(peakutils.indexes(e))  # thres=1, thres_abs=True
        stats['max_peaks']['n'] = len(peakutils.indexes(n))
        stats['max_peaks']['d'] = len(peakutils.indexes(d))
        stats['min_peaks']['e'] = len(peakutils.indexes([(-v) for v in e]))
        stats['min_peaks']['n'] = len(peakutils.indexes([(-v) for v in n]))
        stats['min_peaks']['d'] = len(peakutils.indexes([(-v) for v in d]))
        stats['std_dev']['e'] = np.std(e)
        stats['std_dev']['n'] = np.std(n)
        stats['std_dev']['d'] = np.std(d)
        stats['variance']['e'] = np.var(e)
        stats['variance']['n'] = np.var(n)
        stats['variance']['d'] = np.var(d)
        stats['rms']['e'] = np.sqrt(np.mean([v ** 2 for v in e]))
        stats['rms']['n'] = np.sqrt(np.mean([v ** 2 for v in n]))
        stats['rms']['d'] = np.sqrt(np.mean([v ** 2 for v in d]))
        stats['lng'] = ln
        stats['lat'] = lt
        stats_array.append(copy.deepcopy(stats))

    print('stats array', len(stats_array))
    return stats_array

    # for e, n, d, ln, lt in zip(window_t[0], window_t[1], window_t[2], window_t[3], window_t[4]):
