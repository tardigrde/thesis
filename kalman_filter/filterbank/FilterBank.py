from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver
from kalman_filter.filterbank import kf_utility as util
import numpy as np
from utils import plotter


class FilterBank:
    def __init__(self, types):
        self._types = types
        self.create_filterbank()
        self._adapted_states=[]
        self._adapted_covs=[]

    def create_filterbank(self):
        if self._types == 'all':
            self._cv = KF(dim_x=4, dim_z=4, dim_u=2)
            self._cv_saver = Saver(self._cv)

            self._ca = KF(dim_x=6, dim_z=4, dim_u=2)
            self._ca_saver = Saver(self._ca)

            sigmas_cv = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
            self._ucv = UKF(dim_x=4, dim_z=2, fx=util.f_cv, hx=util.h_cv, dt=1, points=sigmas_cv)
            self._ucv_saver = Saver(self._ucv)

            sigmas_ca = MerweScaledSigmaPoints(4, alpha=.5, beta=2., kappa=1.)
            self._uca = UKF(dim_x=4, dim_z=2, fx=util.f_ca, hx=util.h_ca, dt=0.01, points=sigmas_ca)
            self._uca_saver = Saver(self._uca)

    def set_filters(self, first_measurement):
        util.set_cv_filter(self._cv)
        #util.set_ca_filter(self._ca)
        util.set_ukf_cv_filter(self._ucv)
        util.set_ukf_ca_filter(self._uca)

        # Setting the first state.
        initial_state = np.array([first_measurement.lng, first_measurement.vlng, first_measurement.lat, first_measurement.vlat])
        self._cv.x = self._ucv.x = self._uca.x = initial_state
        # ADJUST THIS
        # self._ca.x = np.array([first_measurement.lng, first_measurement.vlng, .0, first_measurement.lat, first_measurement.vlat, .0])

    def do_adaptive_kf(self, point, max_eps):
        self._cv, self._cv_saver = util.do_cv_k_filtering(point, self._cv, self._cv_saver)
        cv_eps = self._cv.epsilon

        # self._ca, self._ca_saver= util.do_ca_k_filtering(point, self._ca, self._ca_saver)
        # ca_eps = self._ca.epsilon
        self._ca, self._ca_saver = None, None
        ca_eps = None


        self._ucv, self._ucv_saver = util.do_cv_uk_filtering(point, self._ucv, self._ucv_saver)
        ucv_eps = self._ucv.epsilon



        self._uca, self._uca_saver = util.do_ca_uk_filtering(point, self._uca, self._uca_saver)
        uca_eps = self._uca.epsilon

        filters=[self._cv,self._ca , self._ucv, self._uca]
        epsilons=[cv_eps,ca_eps , ucv_eps, uca_eps]
        Q_scale_factor = 10
        # util.adjust_process_noise(filters, epsilons, max_eps, Q_scale_factor)

        adapted_state = util.save_adapted_state(filters,epsilons)
        adapted_state['time'] = point.time
        self._adapted_states.append(adapted_state)


    def results_to_array(self):
        self._cv_saver.to_array()
        # self._ca_saver.to_array()
        self._ucv_saver.to_array()
        self._uca_saver.to_array()

    def smooth_adapted(self, model='ucv'):
        if model == 'ucv':
            xs = np.array([s['x'] for s in self._adapted_states])
            covs = np.array([s['P'] for s in self._adapted_states])

            Ms, P, K = self._ucv.rts_smoother(xs, covs)

            assert len(self._adapted_states) == len(Ms)

            # plotter.plot_rts_output(xs, Ms, range(len(xs)))


            # lng, lat = self.get_smoothed_lng_lat()

            return Ms, P, K


    # def get_smoothed_lng_lat(self):
    #     lng = []
    #     lat = []
    #     for m in self._Ms:
    #         lng.append(m[0])
    #         lat.append(m[2])
    #     return lng, lat




    @property
    def cv_saver(self):
        return self._cv_saver

    @property
    def ca_saver(self):
        return self._ca_saver

    @property
    def ucv_saver(self):
        return self._ucv_saver

    @property
    def uca_saver(self):
        return self._uca_saver

    @property
    def adapted_states(self):
        return self._adapted_states
    # @property
    # def cv(self):
    #     return self._cv
    #
    # @property
    # def ca(self):
    #     return self._ca
    #
    # @property
    # def ucv(self):
    #     return self._ucv
    #
    # @property
    # def uca(self):
    #     return self._uca
