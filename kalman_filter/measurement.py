from kalman_filter import interface


class Measurement:
    def __init__(self, path_imu, path_gps, dir_path):
        self.path_imu = path_imu
        self.path_gps = path_gps
        self.dir_path = dir_path
        self.stats = {}

    def preprocess(self):
        self.acc = interface.get_acceleration_data(self.path_imu)
        self.gps = interface.get_gps_data(self.path_gps)


    def do_kalman_filtering(self):
        self.kalmaned_data_table, self.stats = interface.get_kalmaned_datatable(self.acc, self.gps, self.dir_path)
        pass

    def segment_data(self):
        n = interface.segment_data(self.acc, self.gps)
        # print(len(segmented_z_axis))

    def get_stats(self):
        return self.stats
