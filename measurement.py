from kalman_filter.filterbank.ukf.UnscentedKalmanFilterInterface import UnscentedKalmanFilterInterface
from utils import imu_data_parser, nmea_parser, fuser
from utils import output_creator
from dsp_library import dsp


class Measurement:
    def __init__(self, path_imu, path_gps, dir_path):
        self.path_imu = path_imu
        self.path_gps = path_gps
        self.dir_path = dir_path
        self.stats = {}

    def preprocess(self):
        acc = imu_data_parser.get_imu_dictionary(self.path_imu, data='lists')
        gps = nmea_parser.get_gps_dictionary(self.path_gps, data='lists')
        self.acc, self.gps, self.gps_time_intervals = fuser.trim_and_sync_dataset(acc, gps)
        return self.acc, self.gps

    def get_points(self):
        """
        Must be called after preprocess.
        """
        self.points = fuser.get_points_with_acc(self.acc, self.gps)

    def get_msrmnet_time_intervals(self):
        return self.gps_time_intervals

    def do_kalman_filtering(self):
        kf = UnscentedKalmanFilterInterface('adaptive', self.points, self.dir_path)
        self.kalmaned = kf.do_kalman_filter()
        return self.kalmaned

    def get_potholes(self):
        ts = dsp.get_road_anomaly_timestamps(self.points, self.kalmaned['adapted_states'], self.gps_time_intervals, self.dir_path)
        self.potholes = fuser.map_potholes_back_to_real_world(ts, self.kalmaned['adapted_states'], self.gps_time_intervals)

    def create_KF_outputs(self):
        output_creator.create_outputs(self.dir_path, self.kalmaned)

    def create_PH_outputs(self):
        output_creator.write_potholes_to_shp(self.dir_path, self.potholes)

    def get_acc_gps(self):
        return self.acc, self.gps

    def get_stats(self):
        return self.stats
