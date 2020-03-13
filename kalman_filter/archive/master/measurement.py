from kalman_filter import interface

class Measurement:
    def __init__(self, path_imu, path_gps, metadata):
        self.path_imu = path_imu
        self.path_gps= path_gps
        self.metadata = metadata

    def preprocess(self):
        self.acc = interface.get_acceleration_data(self.path_imu)
        self.gps = interface.get_gps_data(self.path_gps)

    def do_kalman_filtering(self):
        self.kalmaned_data_table = interface.get_kalmaned_datatable(self.acc, self.gps)
        pass

    def segment_data(self):
        pass
        max, min = interface.segment_data(self.acc, self.gps)
        #print(len(segmented_z_axis))
