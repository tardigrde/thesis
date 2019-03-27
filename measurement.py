from kalman_filter.ukf.cv import interfacecv
from kalman_filter.ukf.ca import interfaceca
from dsp_library import dsp
from utils import imu_data_parser, nmea_parser, fuser

from Point import Point



class Measurement:
    def __init__(self, path_imu, path_gps, dir_path):
        self.path_imu = path_imu
        self.path_gps = path_gps
        self.dir_path = dir_path
        self.stats = {}

    def preprocess(self):
        acc = imu_data_parser.get_imu_dictionary(self.path_imu, data='lists')
        gps = nmea_parser.get_gps_dictionary(self.path_gps, data='lists')
        self.acc, self.gps = fuser.trim_and_sync_dataset(acc, gps)
        return self.acc, self.gps

    def get_points(self):
        """
        Must be called after preprocess.
        """
        self.points = fuser.get_points_with_acc(self.acc, self.gps)

    def do_kalman_filtering(self):
        self.kfcv, self.acc_signal = interfacecv.do_ukf(self.dir_path, self.acc, self.gps)
        # kfca = interfaceca.do_ukf_with_acc(self.dir_path, self.acc, self.gps)

        return self.kfcv

    def get_potholes(self):
        n = dsp.get_potholes(self.acc_signal, self.kfcv)
        return n

    def get_acc_gps(self):
        return self.acc, self.gps

    def get_stats(self):
        return self.stats
