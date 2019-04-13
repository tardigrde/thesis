from utils import imu_data_parser
from IMU import IMU

def calculate_calibration_value(path):
    imu = IMU(path)
    pp = imu.preprocessed

# path =r'D:\code\PyCharmProjects\thesis\data\for_calibration\mlt-20190408-220657-138.csv'
path = r'D:\code\PyCharmProjects\thesis\data\20190409\calibration\mlt-20190409-191841-180.csv'
calculate_calibration_value(path)