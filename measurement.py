from kalman_filter.interpolation.one_axis.constant_acceleration import interface
from dsp_library import dsp
from utils import imu_data_parser, nmea_parser


class Measurement:
    def __init__(self, path_imu, path_gps, dir_path):
        self.path_imu = path_imu
        self.path_gps = path_gps
        self.dir_path = dir_path
        self.stats = {}

    def preprocess(self):
        self.acc = imu_data_parser.get_imu_dictionary(self.path_imu, data='lists')
        self.gps = nmea_parser.get_gps_dictionary(self.path_gps,data='lists')

    def get_lists_of_measurements(self):
        acc_lists = self.acc
        gps_lists = self.gps
        return acc_lists, gps_lists

    def do_kalman_filtering(self):
        self.kalmaned_data_table, self.stats = interface.get_kalmaned_datatable(self.acc, self.gps, self.dir_path)
        pass

    # def segment_data(self):
    #     n = interface.segment_data(self.acc, self.gps)
    #     # print(len(segmented_z_axis))

    def get_acc_gps(self):
        return self.acc, self.gps
        # return self.kalmaned_data_table

    def get_filtered_acc(self):
        self.filtered = dsp.do_HPF(self.acc)
        return

    def get_stats(self):
        return self.stats

    def get_state_of(self, state):
        """

        Args:
            state:

        Returns:

        """
        list_of_queried_states = []
        if len(state) > 1:
            for s in state:
                list_of_queried_states.append(self[s])
            return list_of_queried_states
        else:
            return self.state
