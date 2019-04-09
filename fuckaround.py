from utils import imu_data_parser

def calculate_calibration_value(path):
    imu_data_parser.get_imu_dictionary(path)

path =r'D:\code\PyCharmProjects\thesis\data\for_calibration\mlt-20190408-220657-138.csv'
calculate_calibration_value(path)