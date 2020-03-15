from kalman_filter.KalmanFilterInterface import KalmanFilterInterface
from utils import nmea_parser, fuser
from utils import output_creator
from IMU import IMU
from Evaluator import Evaluator


class Measurement:
    def __init__(self, path_imu, path_gps, dir_path, reference_path, invalid_path, szentharomsag_path, max_eps,
                 min_no_of_pothole_like_measurements):
        self.path_imu = path_imu
        self.path_gps = path_gps
        self.dir_path = dir_path
        self.path_to_reference_potholes = reference_path
        self.min_no_of_pothole_like_measurements = min_no_of_pothole_like_measurements
        self.invalid_path = invalid_path
        self.szentharomsag_path = szentharomsag_path
        self.max_eps = max_eps
        self.stats = {}

    def preprocess(self):
        file = '/home/levente/projects/thesis/tests/preprocessed/imu.csv'
        imu = IMU(self.path_imu)
        acc = imu.preprocessed
        write_imu_preprocessed_to_file(file, acc)

        # acc = imu_data_parser.get_imu_dictionary(self.path_imu, data='lists')
        one_level_deep_gps = nmea_parser.get_gps_dictionary(self.path_gps, data='lists')

        file = '/home/levente/projects/thesis/tests/preprocessed/gps.csv'
        #write_gps_preprocessed_to_file(file, one_level_deep_gps)
        self.acc, self.gps, self.gps_time_intervals = fuser.trim_and_sync_dataset(acc, one_level_deep_gps)
        return self.acc, self.gps

    def get_points(self):
        """
        Must be called after preprocess.
        """
        self.points = fuser.get_points_with_acc(self.acc, self.gps)

    def get_msrmnet_time_intervals(self):
        return self.gps_time_intervals

    def do_kalman_filtering(self):
        ukf = KalmanFilterInterface('adaptive', self.points, self.dir_path, self.max_eps)
        ukf.do_kalman_filter()
        self.kalmaned = ukf.filterbank_results

    def evaluate_potholes(self):
        eval = Evaluator(self.path_to_reference_potholes, self.invalid_path, self.szentharomsag_path)
        self.raw_potholes = eval.get_potholes(self.points, self.kalmaned, self.gps_time_intervals,
                                              self.min_no_of_pothole_like_measurements)
        precision = eval.evaluate_potholes()
        on_road_raw_gps = eval.get_kalmaned_on_road(self.result_lists['smoothed'])
        gps = {
            'lng': [i for sublist in self.gps['lng'] for i in sublist],
            'lat': [i for sublist in self.gps['lat'] for i in sublist]

        }
        on_road_surface = eval.get_kalmaned_on_road(gps)
        return precision

    def create_outputs(self, type=None):
        assert type != None
        if type == 'kalmaned':
            self.result_lists = output_creator.create_outputs(self.dir_path, self.kalmaned, type)
        elif type == 'potholes':
            output_creator.create_outputs(self.dir_path, self.raw_potholes, type)

    # def create_PH_outputs(self):
    #     output_creator.write_potholes_to_shp(self.dir_path, self.potholes)

    def get_acc_gps(self):
        return self.acc, self.gps

    def get_stats(self):
        return self.stats


def write_imu_preprocessed_to_file(file, preprocessed):
    import csv

    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        data = preprocessed
        writer.writerow(['time', 'east', 'north', 'down'])
        for t, e, n, d in zip(data.get('_time'), data.get('east'), data.get('north'), data.get('down')):
            writer.writerow([t, e, n, d])


def write_gps_preprocessed_to_file(file, preprocessed):
    import csv

    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        data = preprocessed
        writer.writerow(['time', 'lat', 'lng', 'vlat', 'vlng', 'hdop', 'track'])
        for t, lt, ln, vlt, vln, h, tr in zip(data.get('time'), data.get('lat'), data.get('lng'), data.get('vlt'),
                                              data.get('vln'), data.get('hdop'), data.get('t')):
            writer.writerow([t, lt, ln, vlt, vln, h, tr])
