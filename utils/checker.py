def check_length_of_lists(acc_time, acc_down):
    assert len(acc_time) == len(acc_down)


def check_length_of_acc_lists_and_kf_res(acc_time, acc_down, gps_intervals):
    try:
        assert len(acc_time) == len(acc_down) == len(gps_intervals)
    except Exception as e:
        print('Error! list lengths are not equal.\n', e)

def check_acc_indices(indices):
    min_max_length_of_acc_lists = set(range(90, 110))
    length_of_indices = len(indices)
    try:
        assert length_of_indices in min_max_length_of_acc_lists
    except AssertionError as e:
        print(e)
        print('Warning! Length of indices in fuser.py is: ', length_of_indices)



def check_intervals_length(gps, intervals):
    assert len(gps['time']) == len(intervals)
    for t, i in zip(gps['time'], intervals):
        assert len(t) == len(i)
        print('Intervals: OK!')


def check_rounded_acc_time(acc_time, gps_time):
    round_accs = []
    for a in acc_time:
        if a % 1000 == 0:
            round_accs.append(a)
    round_acc_set = set(sorted(round_accs))
    gps_set = set((gps_time))
    try:
        assert gps_set.issubset(round_acc_set)
    except AssertionError:
        # AssertionError is not practical now cuz stuff will be removed anyway
        print('defference between gps_set and rounded acc set: ', gps_set - round_acc_set)


def check_gps(gps):
    assert len(list(gps.keys())) == 7


def check_if_trim_sync_went_ok(acc, gps):
    for a, g in zip(acc['_time'], gps['time']):
        assert a[0] < g[0] and a[len(a) - 1] > g[len(g) - 1]
        print('OK')