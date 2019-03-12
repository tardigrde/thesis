import numpy as np


def prepare_data_for_batch_kf(acc, gps):
    # for a in acc['time']:
    #     for t, ln, la, hdop in gps
    return acc, gps


def trim_and_sync_dataset(acc_msrmnt, gps_msrmnt):
    # TODO:
    # - if last gps was 5 secs ago, new sublist --> new kalman
    gps_msrmnt = trim_gps(gps_msrmnt)
    acc, gps = trim_and_sync_acc(acc_msrmnt, gps_msrmnt)
    return acc, gps


def trim_gps(gps):
    columns = list(gps.keys())
    count = len(gps['time'])
    five_percent = int(round(count / 10, 0))

    for c in columns:
        if c in gps and len(gps[c]) == count:
            del (gps[c][:five_percent], gps[c][-five_percent:])
        else:
            print('ERROR: trimming gps fails!')
    return gps


def trim_and_sync_acc(acc, gps):
    gps_time = gps['time']
    # TODO:
    # - maybe trim very low speed gps-es
    first_measurement, last_measurement = gps_time[0], gps_time[len(gps_time) - 1]
    at = acc['_time']
    a_first_index, a_last_index = 0, 0

    for i in range(len(at)):
        try:
            if first_measurement >= at[i] and first_measurement <= at[i + 1]:
                a_first_index = i
            elif last_measurement >= at[i] and last_measurement <= at[i + 1]:
                a_last_index = i + 1
        except IndexError as error:
            a_last_index = i
            break
    to_trim = [(a_first_index - 1), (len(at) - a_last_index - 1)]

    columns = list(acc.keys())
    for c in columns:
        del (acc[c][:to_trim[0]], acc[c][-to_trim[1]:])

    acc['_time'] = round_acc_timestamps(acc['_time'], gps_time)
    acc, gps = remove_accleration_where_no_valid_gps(acc, gps)
    return acc, gps

def split_measurement_lists_at_void(data):
    time = data['time'] if 'time' in list(data.keys()) else data['_time']
    last_msrmnt_ts = time[0]
    to_break = [0]
    no_measurment_intervals = []
    for i,t in enumerate(time):
        if t - last_msrmnt_ts > 4999:
            to_break.append(i)
            no_measurment_intervals.append([last_msrmnt_ts, t])
        last_msrmnt_ts = t
    for c in list(data.keys()):
        l = data[c]
        data[c] = [l[i:j] for i, j in zip(to_break, to_break[1:] + [None])]
    return data, no_measurment_intervals


def remove_accleration_where_no_valid_gps(acc, gps):

    gps_time = gps['time']
    acc_time = acc['_time']

    gps,no_measurement_intervals=split_measurement_lists_at_void(gps)

    voidable = []
    for void in no_measurement_intervals:
        for t in range(len(acc_time)):
            if acc_time[t] > void[0] and acc_time[t] < void[1]:
                voidable.append(t)

    for i in reversed(voidable):
        del acc['_time'][i]
        del acc['east'][i]
        del acc['north'][i]
        del acc['down'][i]

    acc, _ = split_measurement_lists_at_void(acc)
    return acc, gps


def round_acc_timestamps(acc_time, gps_time):
    for gt in gps_time:
        last_gps_ts = gt
        for i in range(len(acc_time)):
            try:
                at = acc_time[i]
                at_post = acc_time[i + 1]
                if gt == acc_time[i]:
                    # print('Lucky you. They are the same: ', gt, ' = ', acc_time[i])
                    break
                if gt > acc_time[i] and gt < acc_time[i + 1]:
                    diff_prior = abs(gt - acc_time[i])
                    diff_posterior = abs(gt - acc_time[i + 1])
                    if diff_prior <= diff_posterior:
                        # both if prior is smaller and if prior and posterior are the same, prior should equal the gt's ts
                        acc_time[i] = gt
                    else:
                        acc_time[i + 1] = gt
                    break
            except IndexError as error:
                # print('index out of range: ', error)
                break
    print(gps_time)
    i = []
    for a in acc_time:
        if a % 1000 == 0:
            i.append(a)
    print('acc ', i)
    print(gps_time[0], gps_time[len(gps_time) - 1])
    print(acc_time[0], acc_time[len(acc_time) - 1])
    return acc_time


def interpolate_and_trim_data(acc_lists, gps_lists):
    c = list(acc_lists.keys())

    time, lng, lat, p, v = interpolate_gps_datalists(acc_lists['acc_time'], gps_lists)

    dataset = {
        'time': acc_lists['acc_time'],
        'east': acc_lists['acc_east'],
        'north': acc_lists['acc_north'],
        'down': acc_lists['acc_down'],
        'gps_time': time,
        'lng': lng,
        'lat': lat,
        'hdop': p,
        'vel': v,
    }
    columns = list(dataset.keys())
    count = len(dataset['time'])
    five_percent = int(round(count / 20, 0))

    for c in columns:
        # Checking if all the lists has the same size
        if not len(dataset[c]) == count: return
        count = len(dataset[c])
        del (dataset[c][:five_percent], dataset[c][-five_percent:])

    return dataset


def interpolate_gps_datalists(time, gps_data):
    gps_time = gps_data['time']
    ln = gps_data['ln']
    lt = gps_data['la']
    hdop = gps_data['hdop']
    v = gps_data['v']

    gtime = np.interp(time, gps_time, gps_time).tolist()
    lng = np.interp(time, gps_time, ln).tolist()
    lat = np.interp(time, gps_time, lt).tolist()
    p = np.interp(time, gps_time, hdop).tolist()
    v = np.interp(time, gps_time, v).tolist()

    return gtime, lng, lat, p, v


def _pass_std_devs(acc):
    std_dev_acc_east = np.std(acc['acc_east'])
    std_dev_acc_north = np.std(acc['acc_north'])
    return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}


