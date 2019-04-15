from utils import imu_data_parser
from IMU import IMU
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def calculate_calibration_value(path):
    imu = IMU(path)
    pp = imu.preprocessed


# path =r'D:\code\PyCharmProjects\thesis\data\for_calibration\mlt-20190408-220657-138.csv'
# path = r'D:\code\PyCharmProjects\thesis\data\20190409\calibration\mlt-20190409-191841-180.csv' # calibration
# path =r'D:\code\PyCharmProjects\thesis\data\playground\x\mlt-20190413-190341-11s.csv' # xtengely
# path = r'D:\code\PyCharmProjects\thesis\data\playground\north_south\mlt-20190413-191843-12s.csv' #north_south
# path = r'D:\code\PyCharmProjects\thesis\data\playground\down_up\mlt-20190413-192659-6s.csv' #down_up

# path = r'D:\code\PyCharmProjects\thesis\data\playground\sequence\mlt-20190413-193457-7s.csv' # sequence short
# path = r'D:\code\PyCharmProjects\thesis\data\playground\sequence\mlt-20190413-193513-13s.csv' # sequence long

path = r'D:\code\PyCharmProjects\thesis\data\playground\east-west\mlt-20190413-195521-5s.csv'  # west-east-west-east


# calculate_calibration_value(path)


def read_shape_file(shp_path):
    import fiona
    shapes = fiona.open(shp_path)
    shps = iter(shapes)
    while True:
        try:
            shp = next(shps)
            shp_geom = shape(shp['geometry'])
            print(shp_geom)
            p = Point(734541.035,100812.575)
            print(shp_geom.contains(p))

        except StopIteration:
            print('Last item')
            break

    # print(shape.schema)
    # print('First:', next(iter(shape)))
    # print('Second:', next(iter(shape)))


# shp_path = r'D:\code\PyCharmProjects\thesis\data\real_potholes\potholes.shp' # points
shp_path = r'D:\code\PyCharmProjects\thesis\data\real_potholes\potholes_1m_buffer.shp'  # buffers

read_shape_file(shp_path)
