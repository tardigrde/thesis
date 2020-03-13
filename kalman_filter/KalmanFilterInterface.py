from kalman_filter.filterbank import kf_utility
from .filterbank.FilterBank import FilterBank
import numpy as np


class KalmanFilterInterface:
    def __init__(self, type, points, dir_path,max_eps):
        self._type = type
        self._points = points
        self._dir_path = dir_path
        self._adapted_states = None
        self.max_eps = max_eps

    def __repr__(self):
        pass

    def do_kalman_filter(self):
        points = self._points
        kalam_filter_results = {}
        adapted_states = []

        # set_of_points_to_filter = points[0]

        fb = FilterBank('all')
        fb.create_filterbank()

        for i, set_of_points_to_filter in enumerate(points):
            print('Kalman Filter count: {}'.format(i))

            fb.set_filters(points[i][0])

            for j, p in enumerate(set_of_points_to_filter):
                    fb.do_adaptive_kf(p, self.max_eps)

        fb.results_to_array()

        Xs, _, _ = fb.smooth_adapted()
        self._filterbank_results = {
            'cv': fb.cv_saver,
            'ca': fb.ca_saver,
            'ucv': fb.ucv_saver,
            'uca': fb.uca_saver,
            'adapted': fb.adapted_states,
            'smoothed': Xs,
        }



        # kalam_filter_results = kf_utility.manage_adaptive_filtering(self._type, self._points)
        self._stats = kf_utility.get_stats(self._filterbank_results['adapted'])

    @property
    def filterbank_results(self):
        return self._filterbank_results

    @property
    def adapted_state(self):
        return self._adapted_states

    @property
    def stats(self):
        return self._stats


