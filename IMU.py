from utils import imu_data_parser


class IMU:
    def __init__(self, path):
        self.path = path

    @property
    def preprocessed(self):
        self._dataframe = imu_data_parser._wrangle_data_with_pandas(self.path)
        self._dataframefiltered = imu_data_parser.apply_low_pass(self._dataframe)
        list_of_dicts = imu_data_parser._iterate_through_table_and_do_calculations(self._dataframefiltered)
        return imu_data_parser.pass_acc_dict_of_lists(list_of_dicts)

    @property
    def dataframe(self):
        self._dataframe = imu_data_parser._wrangle_data_with_pandas(self.path)
