from utils import nmea_parser
from utils import imu_data_parser


def get_length_of_gps_file(path):
    data = nmea_parser._read_log_file(path)
    print(data[0])
    return len(data)


def get_length_of_gps_lists(gps_measurements):
    no_of_samples = len(gps_measurements['time'])
    for i in list(gps_measurements.keys()):
        if len(gps_measurements[i]) != no_of_samples:
            return False
    return no_of_samples


def test_nmea_timestamps_precision(time_measurements):
    first_ts = time_measurements[0]
    last_ts = first_ts - 1000
    for time in time_measurements:
        if time - 1000 != last_ts:
            return False
        last_ts = time
    return True


def get_length_of_imu_lists(imu_lists):
    no_of_samples = len(imu_lists['acc_time'])
    for i in list(imu_lists.keys()):
        if len(imu_lists[i]) != no_of_samples:
            return False
    return no_of_samples

def get_length_of_imu_csv(path):
    imu_df = imu_data_parser._wrangle_data_with_pandas(path)
    return len(imu_df)

def test_imu_timestamps_precision(time_measurements):
    first_ts = time_measurements[0]
    last_ts = first_ts - 10
    for time in time_measurements:
        if time - 10 < last_ts:
            return False
        last_ts = time
    return True
