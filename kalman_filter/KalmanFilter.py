from kalman_filter.filterbank.ukf import ukf_interface

class KalmanFilter:
    def __init__(self, type, points, dir_path):
        self._type=type
        self._points = points
        self._dir_path = dir_path
        pass

    def __repr__(self):
        pass

    def do_kalman_filter(self):
        ukf_interface.do_unscented_kalman_filtering(self._type, self._points, self._dir_path)




