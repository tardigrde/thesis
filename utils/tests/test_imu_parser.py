from utils import imu_data_parser
from utils.tests import test_functions as tf


def test_length_of_input_output(len_in, len_out):
     if not len_in == len_out:
        return False
     return True


def test_range_of_timestamp(is_precise):
    if not is_precise:
        return False
    return True


def test_range_of_acc(axis):
    for a in axis:
        if a > 2 or a < -2:
            return False
    return True

def test_imu_parser(path):
    import time
    start_time = time.time()

    # get the data
    imu_measurements = imu_data_parser.get_imu_dictionary(path, 'lists')
    results = []

    # Test 1: Test length of measurement and prepeared data.
    length_of_log_file = tf.get_length_of_imu_csv(path)
    length_of_lists = tf.get_length_of_imu_lists(imu_measurements)

    one = test_length_of_input_output(length_of_log_file, length_of_lists)
    results.append(one)

    # Test 2: Test timestamps.
    ts_precision = tf.test_imu_timestamps_precision(imu_measurements['acc_time'])

    two = test_range_of_timestamp(ts_precision)
    results.append(two)
    to_increment = 0

    for r in results:
        to_increment = to_increment + 1
        if not r:
            return ('IMU tests failed, particularly the ' + str(to_increment) + 'th one.')
    return ('NMEA tests: OK!')

    print("--- Testing imu_parser took %s seconds ---" % (time.time() - start_time))


path = r'D:\PyCharmProjects\thesis\tests\fixtures\fixtures.csv'
text = test_imu_parser(path)
print(text)
