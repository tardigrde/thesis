from kalman_filter.filterbank.ukf import ukf_utility

class UnscentedKalmanFilterInterface:
    def __init__(self, type, points, dir_path):
        self._type=type
        self._points = points
        self._dir_path = dir_path
        self._adapted_state = None
        pass

    def __repr__(self):
        pass

    def do_kalman_filter(self):
        return ukf_utility.do_unscented_kalman_filtering(self._type, self._points, self._dir_path)

    @property
    def adapted_state(self):
        pass




