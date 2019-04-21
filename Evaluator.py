from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.geometry import shape
import fiona
from dsp_library import dsp
from utils import fuser, checker


class Evaluator:
    def __init__(self, path_to_reference_shape):
        self.path_to_reference_shape = path_to_reference_shape

    # @property
    # def potholes(self):
    #     return self._potholes

    def get_potholes(self, points, kalmaned, gps_time_intervals,min_no_of_pothole_like_measurements):
        pothole_ts = self.get_pothole_timespans(points, gps_time_intervals)
        self.raw_potholes = fuser.map_potholes_back_to_real_world(pothole_ts, kalmaned, gps_time_intervals, min_no_of_pothole_like_measurements)
        return self.raw_potholes

    def get_pothole_timespans(self, points, gps_time_intervals):
        acc_time = fuser.get_acc_axis(points, '_time', gps_time_intervals)
        acc_down = fuser.get_acc_axis(points, 'down', gps_time_intervals)

        checker.check_length_of_lists(acc_time, acc_down)

        potholes, acc_time_windows = self.classify_windows(acc_time, acc_down)
        pothole_ts = fuser.map_potholes_to_timestamp(potholes, acc_time_windows)
        return pothole_ts

    def classify_windows(self, acc_time, acc_down):
        acc_time_windows = dsp.do_windowing(acc_time)

        filtered = dsp.do_low_pass_filter(acc_down)
        filtered_acc_down_windows = dsp.do_windowing(filtered, hamming=True)

        potholes = dsp.do_classification(acc_time_windows, filtered_acc_down_windows)

        # plotter.plot_ned_acc(fig_dir, t, down_acc)
        return potholes, acc_time_windows

    def evaluate_potholes(self):
        shapes = self.read_shape_file(self.path_to_reference_shape)
        ph = self.raw_potholes
        true_positives = []
        false_positives = []
        false_negatives = []
        shps = iter(shapes)
        while True:
            try:
                shp = next(shps)
                shp_geom = shape(shp['geometry'])
                # print(shp_geom)
                for ln, lt, time, llh, eps in zip(ph['lng'], ph['lat'], ph['time'], ph['llh'], ph['probability']):
                    point_geom = Point(ln, lt)
                    pthl = {
                        'point': point_geom,
                        'time': time,
                        'llh': llh,
                        'epsilon': eps
                    }
                    if shp_geom.contains(point_geom):
                        true_positives.append(pthl)
                    else:
                        false_positives.append(pthl)
            except StopIteration:
                print('Last item')
                break
        print(
            'Total count of detectedpotholes was:',len(ph['time']),
            '.\n From that ',len(true_positives),' was true positive, and ',
            len(false_positives),' was false positives.!'
        )

    def read_shape_file(self, shp_path):
        shapes = fiona.open(shp_path)
        return shapes
