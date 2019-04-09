from __future__ import division
from scipy.signal import find_peaks, butter, spectrogram
from utils import fuser
from utils import plotter
import pandas as pd
import peakutils
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import time


def get_road_anomaly_timestamps(points, adaptive_kf_result, time_intervals, dir_path):
    pothole_timestamps= classify_windows(points, adaptive_kf_result, time_intervals, dir_path)
    # write_kalmaned_data_to_file(adaptive_kf_result)
    # write_interval_data_to_file(time_intervals)
    return pothole_timestamps


def classify_windows(points, adaptive_kf_result, time_intervals, dir_path):
    start_time = time.time()
    acc_time, acc_down, kf_res = prepare_data_for_filtering(points, adaptive_kf_result)
    filtered = do_low_pass_filter(acc_down)
    acc_time_windows = do_windowing(acc_time)
    filtered_acc_down_windows = do_windowing(filtered, hamming=True)
    potholes = do_classification(acc_time_windows, filtered_acc_down_windows)

    # Concept:
    #   we gonna get the time windows, add them al ltogether, transform it to a set
    #
    pothole_timestamps = map_potholes_to_timestamp(acc_time_windows, potholes)

    # get_frequencies(filtered)
    #
    # windows = {
    #     "_time": do_windowing(acc_time, hamming=True),
    #     "down": do_windowing(acc_down),
    # }

    # ffted_windows = [do_fft(down_subset) for down_subset in windows['down']]
    # windowed_time = windows['_time']
    # print(len(windowed_time), len(windowed_time[0]))

    # down_subsets_filtered = [do_HPF(down_subset) for down_subset in windows['down']]
    # st = calculate_stats(windows)
    # plotter.plot_ned_acc(fig_dir, t, down_acc)
    print("--- Segmentation took %s seconds ---" % (time.time() - start_time))
    return pothole_timestamps


def do_classification(acc_time_windows, acc_down_windows):
    if not len(acc_down_windows) == len(acc_time_windows):
        print('down and time are not the same lenght')
        return
    potholes = {
        'thresh': [],
        'diff': [],
        'std': [],
        'combined': []
    }
    for i, down_window in enumerate(acc_down_windows):
        # plt.plot(down_window)
        # plt.show()

        thresh = classify_based_on_threshold(i, down_window)
        if thresh:
            potholes['thresh'].append(thresh)

        diff = classify_based_on_absolut_difference(i, down_window)
        if diff:
            potholes['diff'].append(diff)

        std = classify_based_on_std_dev(i, down_window)
        if std:
            potholes['std'].append(std)
    potholes['combined'] = find_identical_indices(potholes)

    return potholes


def find_identical_indices(potholes):
    list_of_lists = []
    list_of_lists.append(potholes['thresh'])
    list_of_lists.append(potholes['diff'])
    list_of_lists.append(potholes['std'])
    intsect = set.intersection(*[set(list) for list in list_of_lists])
    # seen = set()
    # repeated = set()
    #
    # for list in list_of_lists:
    #     for i in set(list):
    #         if i in seen:
    #             repeated.add(i)
    #         else:
    #             seen.add(i)
    return intsect


def classify_based_on_threshold(i, down_window):
    min_threshold, max_threshold = (-0.08, 0.08)
    indices_in_window = []
    for j, down in enumerate(down_window):
        if min_threshold < down < max_threshold:
            pass
        else:
            indices_in_window.append(j)
    # Change this if in doubt
    if len(indices_in_window) >= 5:
        return i
    else:
        return None


def classify_based_on_absolut_difference(i, down_window):
    max_val = max(down_window)
    min_val = min(down_window)
    diff = max_val - min_val
    if diff > 0.16:
        return i
    else:
        return None


def classify_based_on_std_dev(i, down_window):
    std = np.std(down_window)
    if std > 0.04:
        return i
    else:
        return None


def map_potholes_to_timestamp(acc_time, potholes):
    indices = sorted(list(potholes['combined']))
    timestamp_lists = [acc_time[index] for index in indices]
    flat_timestamps = [t for list in timestamp_lists for t in list]
    set_of_timestamps = set(flat_timestamps)
    return list(set_of_timestamps)


# def get_low_pass_filtered_data(acc_time, acc_down, kf_res):
#
#     filtered = do_low_pass_filter(acc_down)
#     check_length_of_lists(acc_time, filtered)
#
#     # write_acc_data_to_file(acc_time, filtered)
#
#
#     return filtered


def get_frequencies(filtered):
    f, t, Sxx = spectrogram(np.array(filtered), fs=100, window='hann', nperseg=100, noverlap=25)
    # plt.pcolormesh(t, f, Sxx)
    # plt.show()
    # return spectroed


def prepare_data_for_filtering(points, kf_res):
    acc_time_per_point = [p.acc['_time'] for list in points for p in list]
    acc_down_per_point = [p.acc['down'] for list in points for p in list]

    check_length_of_acc_lists_and_kf_res(acc_time_per_point, acc_down_per_point, kf_res)

    acc_time = [elem for list in acc_time_per_point for elem in list]
    acc_down = [elem for list in acc_down_per_point for elem in list]

    check_length_of_lists(acc_time, acc_down)

    return acc_time, acc_down, kf_res


def do_windowing(axis, hamming=False, window_size=100):
    n = window_size
    m = round(n / 4)  # overlap_size
    if hamming == False:
        windows = [axis[i:i + n] for i in range(0, len(axis), n - m)]
    else:
        windows = [do_hamming(axis[i:i + n], n) for i in range(0, len(axis), n - m)]
        # windows = [do_low_pass_filter(axis[i:i + n], n) for i in range(0, len(axis), n - m)]
    return windows


def do_hamming(dataset, n):
    window = np.hamming(n)
    if len(dataset) != n: window = np.hamming(len(dataset))
    windowed_data_og = np.multiply(dataset, window)
    return windowed_data_og


def do_low_pass_filter(acc_down):
    """
    https://plot.ly/python/fft-filters/
    Args:
        acc_down:

    Returns:

    """

    fc = .1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = .08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute a low-pass filter.
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = np.blackman(N)  # 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    filtered = np.convolve(acc_down, sinc_func)
    # TODO:
    # - interpolate
    # - make HPF work
    # if not len(filtered) == len(acc_down) and len(filtered) > len(acc_down):
    #     print('OOOOOOOOOOOPS', len(filtered), len(acc_down))
    #     diff = len(filtered) - len(acc_down)
    #     print(diff)
    #     # filtered_windows_cut =np.delete(filtered, [i for i in range(diff)])
    #     del (list(filtered)[-diff:])
    #     print('len of og: ', len(acc_down), 'len of filtered after deletion: ',len(filtered))
    acc_down, trimmed_filtered = sync_og_and_filtered(acc_down, filtered)

    # plt.subplot(211)
    # plt.plot(acc_down)
    # plt.subplot(212)
    # plt.plot(trimmed_filtered)

    # plt.show()

    plt.plot(acc_down, color='red')
    plt.plot(trimmed_filtered, color='green')
    plt.title("Hamming windows filtered")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()

    return trimmed_filtered


def sync_og_and_filtered(acc_down, filtered):
    diff = int((len(filtered) - len(acc_down)) / 2)
    trimmed = list(filtered)
    del (trimmed[:diff], trimmed[-diff:])
    synced = check_length_of_filtered(acc_down, trimmed)

    if not synced:
        diff = abs(len(trimmed) - len(acc_down))
        if len(trimmed) > len(acc_down):
            del (trimmed[:diff], trimmed[-diff:])
        else:
            del (acc_down[:diff], acc_down[-diff:])
    check_length_of_filtered(acc_down, trimmed)
    # check_length_of_lists(acc_time, filtered)
    return acc_down, trimmed


def check_length_of_filtered(acc_down, trimmed_list):
    difference_between_length = abs(len(acc_down) - len(trimmed_list))
    try:
        assert difference_between_length == 0
        return True
    except:
        print('Length mismatch after low pass filter')
        return False


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
    length_of_window = len(window)

    # Sampling rate and time vector
    start_time = 0  # seconds
    end_time = length_of_window / 100  # seconds
    sampling_rate = 100  # Hz
    N = (end_time - start_time) * sampling_rate  # array size

    # Vibration data generation
    time = np.linspace(start_time, end_time, round(N))

    plt.figure(1)
    plt.xlabel('Time')
    plt.ylabel('Vibration (g)')
    plt.title('Time Domain')
    plt.plot(time, window, 'green')

    # Nyquist Sampling Criteria
    T = 1 / sampling_rate  # inverse of the sampling rate
    x = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    # FFT algorithm
    yr = fft(list(window))  # "raw" FFT with both + and - frequencies
    y = 2 / N * np.abs(yr[0:np.int(N / 2)])  # positive freqs only

    # Plotting the results
    plt.figure(2)
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('????')
    plt.plot(x, y, 'yellow')

    iffted_res = ifft(yr)
    # We here want to have the original acc data seen corresponding to ifft res
    plt.figure(3)
    plt.plot(time, iffted_res, 'r')
    plt.plot(time, window, 'b')
    plt.show()

    return iffted_res


def find_max_peaks(z_axis, height):
    peaks, _ = find_peaks(z_axis, height)
    return peaks, _


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


def write_interval_data_to_file(time_intervals):
    time_intervals = [i for lists in time_intervals for i in lists]
    with open(r'D:\code\PyCharmProjects\thesis\data\trolli_playground\kalmaned_data\interval_data.csv', 'w') as outfile:
        for i in time_intervals:
            line_to_write = str(i[0]) + ',' + str(i[1]) + '\n'
            outfile.write(line_to_write)


def write_kalmaned_data_to_file(adaptive_kf_result):
    with open(r'D:\code\PyCharmProjects\thesis\data\trolli_playground\kalmaned_data\kalmaned_data.csv', 'w') as outfile:
        for t, p in adaptive_kf_result:
            line_to_write = str(t) + ',' + str(p[0]) + ',' + str(p[1]) + '\n'
            outfile.write(line_to_write)


def write_acc_data_to_file(acc_time, filtered):
    accs = []
    with open(r'D:\code\PyCharmProjects\thesis\data\trolli_playground\acc_data\acc_data.csv', 'w') as outfile:
        for t, d in zip(acc_time, filtered):
            line_to_write = str(t) + ',' + str(d) + '\n'
            outfile.write(line_to_write)
    for t, d in zip(acc_time, filtered):
        accs.append({t: d})


def check_length_of_lists(acc_time, acc_down):
    assert len(acc_time) == len(acc_down)


def check_length_of_acc_lists_and_kf_res(acc_time, acc_down, kf_res):
    try:
        assert len(acc_time) == len(acc_down) == len(kf_res)
    except Exception as e:
        print('Error! list lengths are not equal.\n', e)
