from kalman_filter.filterbank.ukf import ukf_utility


class UnscentedKalmanFilterInterface:
    def __init__(self, type, points, dir_path):
        self._type = type
        self._points = points
        self._dir_path = dir_path
        self._adapted_states = None
        pass

    def __repr__(self):
        pass

    def do_kalman_filter(self):

        kalam_filter_results=ukf_utility.do_unscented_kalman_filtering(self._type, self._points, self._dir_path)
        self._adapted_states = kalam_filter_results['adapted_states']
        self._stats = self.get_stats(self._adapted_states)
        return kalam_filter_results

    @property
    def adapted_state(self):
        return self._adapted_states

    @property
    def stats(self):
        return self._stats

    def get_stats(self, adapted_states):
        types = []
        epsilons = []
        below_three= []
        sum_of_epsilons = .0
        for state in adapted_states:
            epsilon = state[1][2]
            # epsilons.append(epsilon)

            type = state[1][3]
            types.append(type)
            sum_of_epsilons += epsilon
            if epsilon < 3:
                below_three.append(epsilon)
        cv = types.count(0)
        ca = types.count(1)
        print('CVd: {}, CAd: {}'.format(cv, ca))
        print('Length of epsilons below three: ', len(below_three))
        print('The sum of epsilons are:', sum_of_epsilons)

