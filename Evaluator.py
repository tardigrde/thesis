from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.geometry import shape
import fiona
from dsp_library import dsp
from utils import fuser, checker


class Evaluator:
    def __init__(self, path_to_reference_shape, invalid_path, szentharomsag_path):
        self.path_to_reference_shape = path_to_reference_shape
        self.invalid_path = invalid_path
        self.valid_pothole = None
        self.szentharomsag_path = szentharomsag_path

    # @property
    # def potholes(self):
    #     return self._potholes

    def get_potholes(self, points, kalmaned, gps_time_intervals, min_no_of_pothole_like_measurements):
        pothole_ts = self.get_pothole_timespans(points, gps_time_intervals)
        self.raw_potholes = fuser.map_potholes_back_to_real_world(pothole_ts, kalmaned, gps_time_intervals,
                                                                  min_no_of_pothole_like_measurements)
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
        shapes = read_shape_file(self.path_to_reference_shape)
        potholes = self.remove_invalid_potholes(self.raw_potholes)
        potholes_found_len = len(potholes)

        precisions = []
        true_positives = []
        already_marked = []
        shps = iter(shapes)
        while True:
            try:
                count = 0
                shp = next(shps)
                shp_geom = shape(shp['geometry'])
                for i, ph in enumerate(potholes):
                    point_geom = ph['point']
                    if shp_geom.contains(point_geom):
                        already_marked.append(i)
                        pthl = {
                            'point': point_geom,
                            'time': ph['time'],
                            'llh': ph['llh'],
                            'prob': ph['prob']
                        }
                        true_positives.append(pthl)
                        count += 1
                precisions.append(count)
                for marked in already_marked:
                    del potholes[marked]
                already_marked = []
            except StopIteration:
                print('Last item')
                break

        true_pos_len = len(true_positives)
        print(
            'Found potholes was:{}.\n  true positive: {},\n false positives: {}'.format(
                potholes_found_len, true_pos_len, potholes_found_len - true_pos_len))
        return precisions

    def remove_invalid_potholes(self, ph):
        shapes = read_shape_file(self.invalid_path)

        invalid_potholes = []
        for i, (ln, lt, time, llh, prob) in enumerate(
                zip(ph['lng'], ph['lat'], ph['time'], ph['llh'], ph['probability'])):
            point_geom = Point(ln, lt)
            pthl = {
                'point': point_geom,
                'time': time,
                'llh': llh,
                'prob': prob
            }
            shps = iter(shapes)
            self.valid_pothole = None
            self.get_validated_pothole(shps, point_geom, pthl)
            if self.valid_pothole is not None:
                invalid_potholes.append(self.valid_pothole)

        print('{} potholes are on non-measureable area.'.format(len(ph['time']) - len(invalid_potholes)))
        return invalid_potholes

    def remove_invalid_kalmaned(self, ph):
        shapes = read_shape_file(self.invalid_path)

        valid_points = []
        for i, (ln, lt) in enumerate(
                zip(ph['lng'], ph['lat'])):
            point_geom = Point(ln, lt)
            pthl = {
                'lng': ln,
                'lat': lt,
            }
            shps = iter(shapes)
            self.valid_pothole = None
            self.get_validated_pothole(shps, point_geom, pthl)
            if self.valid_pothole is not None:
                valid_points.append(self.valid_pothole)

        return valid_points

    def get_kalmaned_on_road(self, kalmaned):

        szentharomsag = read_shape_file(self.szentharomsag_path)
        shp_geom = shape(szentharomsag[0]['geometry'])

        kalmaned= self.remove_invalid_kalmaned(kalmaned)

        on_road = []

        for i, k in enumerate(kalmaned):
            point_geom = Point(k['lng'], k['lat'])
            point = {
                'point': point_geom,
                'id':i,
            }
            if shp_geom.contains(point_geom):
                on_road.append(point)
        time = len(kalmaned)
        positive_count = len(on_road)
        print('OG count: {},\non road: {},\noutside:{},\npercent:{}'.
              format(time, positive_count, time - positive_count, positive_count * 100 / time))

        return on_road


    def get_validated_pothole(self, shps, point_geom, pthl):
        try:
            shp = next(shps)
            shp_geom = shape(shp['geometry'])
            if not shp_geom.contains(point_geom):
                self.valid_pothole = pthl
                self.get_validated_pothole(shps, point_geom, pthl)
            else:
                self.valid_pothole = None
        except StopIteration:
            validated = self.valid_pothole


def read_shape_file(shp_path):
    shapes = fiona.open(shp_path)
    return shapes

# def remove_invalid_potholes(self, ph):
#     shapes = self.read_shape_file(self.invalid_path)
#     shps = iter(shapes)
#     potholes_without_invalid = []
#     while True:
#         try:
#             shp = next(shps)
#             shp_geom = shape(shp['geometry'])
#             for i, (ln, lt, time, llh, prob) in enumerate(
#                     zip(ph['lng'], ph['lat'], ph['time'], ph['llh'], ph['probability'])):
#                 point_geom = Point(ln, lt)
#                 pthl = {
#                     'point': point_geom,
#                     'time': time,
#                     'llh': llh,
#                     'prob': prob
#                 }
#
#                 if not shp_geom.contains(point_geom):
#                     potholes_without_invalid.append(pthl)
#
#         except StopIteration:
#             break
#
#     print('{} potholes are on non-measureable area.'.format(len(ph['time']) - len(potholes_without_invalid)))
#     return potholes_without_invalid
