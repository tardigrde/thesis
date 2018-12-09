from pyproj import Proj, transform
from gps_data import GPS_data
from imu_data import IMU_data
from kalman import Kalman
from initial_params import Initial_params
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


class Interface:
    def __init__(self, in_imu_file, in_gps_file):

        self.in_imu = in_imu_file
        self.in_gps = in_gps_file
        self.attributes = {}
        self.sorted_timestamps = []

    def get_acceleration_data(self):
        """
        Instanciate a set of measurements and transform it to a dictionary of absolute acceleration data.
        Input: filename.
        Output: dictionary of absoulte acceleration data.
        """
        imu = IMU_data(self.in_imu)

        trimmed_dataframe = imu.trim_datatable()

        measurement_dictionary = imu.extract_columns(trimmed_dataframe)

        quaternion_list = imu.get_quaternion(measurement_dictionary)

        rotation_matrix_list = imu.get_rotation_matrix(quaternion_list)

        absolute_acceleration_list = imu.get_absolute_acc(rotation_matrix_list)

        true_absolute_acceleration_dictionary = imu.get_true_abs_acc_dict(absolute_acceleration_list)

        return true_absolute_acceleration_dictionary

    def get_gps_data(self):
        gps = GPS_data(self.in_gps)

        gps_dictionary = gps.get_gps_dictionary()

        return gps_dictionary

    def pass_gps_list(self, gps):
        time, ln, la, vla, vln, hdop = [], [], [], [], [], []
        timestamps = sorted(list(gps.keys()))
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:23700')
        for t in timestamps:
            values = gps[t]
            # ln.append(values['lng'])
            # la.append(values['lat'])
            vln.append(values['vlng'])
            vla.append(values['vlat'])
            hdop.append(values['hdop'])
            time.append(values['time'])
            lng, lat =  transform(inProj,outProj,values['lng'],values['lat'])
            ln.append(lng)
            la.append(lat)

        return {'ln': ln, 'la': la, 'vln': vln, 'vla': vla, 'hdop': hdop, 'time': time}
        # data = [np.asarray(latlong), og_lat, og_lng]
        # return data

    def pass_acc_list(self, acc):
        timestamps = sorted(list(acc.keys()))
        acc_time, acc_east, acc_north, acc_down = [], [], [], []
        for t in timestamps:
            values = acc[t]
            acc_east.append(values['acc_east'])
            acc_north.append(values['acc_north'])
            acc_down.append(values['acc_down'])
            acc_time.append(values['time'])
        return {'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down, 'acc_time': acc_time}

    # def _pass_std_devs(self, acc_east, acc_north, gps_lng, gps_lat, gps_vlng, gps_vlat):
    #
    #     std_dev_ln = np.std(gps_lng)
    #     std_dev_la = np.std(gps_lat)
    #     std_dev_vln = np.std(gps_vlng)
    #     std_dev_vla = np.std(gps_vlat)
    #     std_dev_acc_east = np.std(acc_east)
    #     std_dev_acc_north = np.std(acc_north)
    #
    #     return {'ln': std_dev_ln, 'la': std_dev_la, 'vln': std_dev_vln, 'vla': std_dev_vla,
    #             'acc_east': std_dev_acc_east, 'acc_north': std_dev_acc_north}

    def _pass_std_devs(self, acc_east, acc_north):
        std_dev_acc_east = np.std(acc_east)
        std_dev_acc_north = np.std(acc_north)

        return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}

    def interpolate_gps_data(self, acc, gps):

        gps_lists = self.pass_gps_list(gps)
        acc_lists = self.pass_acc_list(acc)

        acc_time = acc_lists['acc_time']
        acc_east = acc_lists['acc_east']
        acc_north = acc_lists['acc_north']
        acc_down = acc_lists['acc_down']

        gps_time = gps_lists['time']
        gps_lng = gps_lists['ln']
        gps_lat = gps_lists['la']
        gps_vln = gps_lists['vln']
        gps_vla = gps_lists['vla']
        gps_hdop = gps_lists['hdop']

        interpolated_lng = np.interp(acc_time, gps_time, gps_lng).tolist()
        interpolated_lat = np.interp(acc_time, gps_time, gps_lat).tolist()
        interpolated_vlng = np.interp(acc_time, gps_time, gps_vln).tolist()
        interpolated_vlat = np.interp(acc_time, gps_time, gps_vla).tolist()
        interpolated_hdop = np.interp(acc_time, gps_time, gps_hdop).tolist()

        if (len(acc_time) == len(acc_down) == len(acc_east) == len(acc_north) == len(interpolated_lat) == len(
                interpolated_lng) == len(interpolated_vlat) == len(interpolated_vlng) == len(interpolated_hdop)):
            return {'acc_time': acc_time, 'lng': interpolated_lng, 'lat': interpolated_lat, 'vlng': interpolated_vlng,
                    'vlat': interpolated_vlat, 'hdop': interpolated_hdop, 'acc_east': acc_east, 'acc_north': acc_north,
                    'acc_down': acc_down}
        else:
            print('NEM')

    def _predict(self, X_minus, P_minus, F, Q, B, std_devs, acc_east, acc_north):
        kalman = Kalman()
        dt = 1
        Q[0, 0] = (std_devs['std_dev_acc_east'] * dt * dt / 2)**2
        Q[1, 1] = (std_devs['std_dev_acc_north'] * dt * dt / 2)**2
        Q[2, 2] = (std_devs['std_dev_acc_east'] * dt)**2
        Q[3, 3] = (std_devs['std_dev_acc_north'] * dt)**2
        Q[0, 2] = (std_devs['std_dev_acc_east'] * dt * dt / 2) * (std_devs['std_dev_acc_east'] * dt)
        Q[1, 3] = (std_devs['std_dev_acc_north'] * dt * dt / 2) * (std_devs['std_dev_acc_north'] * dt)

        U = np.transpose([acc_east, acc_north])

        #print('Q: {} and U: {}'.format(Q, U))

        predict = kalman.kf_predict(X_minus, P_minus, F, Q, B, U)
        return predict

    def _update(self, X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat):
        kalman = Kalman()
        R[0][0] = hdop * hdop
        R[1][1] = hdop * hdop
        R[2][2] = hdop * hdop
        R[3][3] = hdop * hdop
        #print('R is {}'.format(R))
        X = np.transpose([lng, lat, vlng, vlat])
        #print('X is {}'.format(X))
        Y = X * H + R
        #print('Y is {}'.format(Y))

        update = kalman.kf_update(X_minus, P_minus, Y, H, R)

        return update

    def do_pykalmaning(self, acc, gps):

        init = Initial_params()
        init_params = init.get_initial_parameters()
        #print(init_params)
        P = init_params['P']
        H = init_params['H']
        R = init_params['R']
        Q = init_params['Q']
        F = init_params['F']
        B = init_params['B']

        interpolated_attribute_table = self.interpolate_gps_data(acc, gps)
        acc_time = interpolated_attribute_table['acc_time']
        acc_east = interpolated_attribute_table['acc_east']
        acc_north = interpolated_attribute_table['acc_north']
        gps_lng = interpolated_attribute_table['lng']
        gps_lat = interpolated_attribute_table['lat']
        gps_vlng = interpolated_attribute_table['vlng']
        gps_vlat = interpolated_attribute_table['vlat']
        gps_hdop = interpolated_attribute_table['hdop']

        std_devs = self._pass_std_devs(acc_east, acc_north)

        updated = []
        predicted = []
        lng_to_plot = []
        lat_to_plot = []
        og_lng, og_lat = [], []

        counter = 0
        for time, a_east, a_north, lat, lng, vlat, vlng, hdop in zip(acc_time, acc_east, acc_north, gps_lat, gps_lng, gps_vlat, gps_vlng, gps_hdop):

            og_lng.append(lng)
            og_lat.append(lat)

            if counter == 0:
                X_minus = np.transpose([lng, lat, vlng, vlat])
                P_minus = P
                counter = counter + 1
            else:
                X_minus = updated[len(updated) - 1]['X']
                P_minus = updated[len(updated) - 1]['P']

            predict = self._predict(X_minus, P_minus, F, Q, B, std_devs, a_east, a_north) # 1
            #print('Predicted STATE is {}'.format(predict['X']))
            predicted.append(predict)

            X_minus = predicted[len(predicted) - 1]['X']
            P_minus = predicted[len(predicted) - 1]['P'] #2,3

            update = self._update(X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat)
            updated.append(update)
            #print('Updated STATE is {}'.format(update['X']))

            lng_to_plot.append(update['X'][0][0])
            lat_to_plot.append(update['X'][1][1])
        plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'r--')
        plt.show()
        out = './teszt/roszke-szeged/kalmaned_coordinates1.csv'
        with open(out, 'w') as out:
            for ln, la in zip(lng_to_plot, lat_to_plot):
                out.write(str(ln) + ',' + str(la) + '\n')






    # def _predict(self, X_minus, P_minus, F, Q, B, std_devs, acc_east, acc_north):
    #     kalman = Kalman()
    #     dt = 1
    #     Q[0,0] = std_devs['acc_east'] * dt* dt
    #     Q[1,1] = std_devs['acc_north'] * dt* dt
    #     Q[2,2] = std_devs['acc_east'] * dt
    #     Q[3,3] = std_devs['acc_north'] * dt
    #     """
    #     Q[0,2] = std_devs['ln'] * std_devs['la']
    #     Q[1,3] = std_devs['vln'] * std_devs['vla']
    #     """
    #
    #     U = np.transpose([acc_north, acc_east])
    #     predict = kalman.kf_predict(X_minus, P_minus, F, Q, B, U)
    #     return predict
    #
    # def _update(self, X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat):
    #     kalman = Kalman()
    #     R[0,0] = hdop * hdop
    #     R[1,1] = hdop * hdop
    #     R[2,2] = hdop
    #     R[3,3] = hdop
    #     X = np.transpose([lng, lat])
    #     Y = X * H + R
    #     print('LEN!')
    #     print(len(Y))
    #
    #     update = kalman.kf_update(X_minus, P_minus, Y, H, R)
    #
    #     return update
# def get_kalmaned_coordinates(self, gps, acc):
#
#
#     interpolated_attribute_table = self.interpolate_gps_data(acc, gps)
#     acc_time = interpolated_attribute_table['acc_time']
#     acc_east = interpolated_attribute_table['acc_east']
#     acc_north = interpolated_attribute_table['acc_north']
#     gps_lng = interpolated_attribute_table['lng']
#     gps_lat = interpolated_attribute_table['lat']
#     gps_vlng = interpolated_attribute_table['vlng']
#     gps_vlat = interpolated_attribute_table['vlat']
#     gps_hdop = interpolated_attribute_table['hdop']
#
#     init = Initial_params()
#     init_params = init.get_initial_parameters()
#     P = init_params['P']
#     H = init_params['H']
#     R = init_params['R']
#     Q = init_params['Q']
#     F = init_params['F']
#     B = init_params['B']
#
#     std_devs = self._pass_std_devs(acc_east, acc_north, gps_lng, gps_lat, gps_vlng, gps_vlat)
#
#
#     updated = []
#     predicted = []
#     lng_to_plot = []
#     lat_to_plot = []
#     og_lng = []
#     og_lat = []
#
#     counter = 0
#     for time, acc_east, acc_north, lng, lat, vlng, vlat, hdop in zip(acc_time, acc_east, acc_north, gps_lng, gps_lat, gps_vlng, gps_vlat, gps_hdop):
#
#         #time, acc_east, acc_north, lng, lat, vlng, vlat, hdop = point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7]
#         og_lng.append(lng)
#         og_lat.append(lat)
#
#         if counter == 0:
#             X_minus = np.transpose([lng,lat,vlng,vlat])
#             P_minus = P
#             counter = counter + 1
#         else:
#             last_X = updated[len(updated) - 1]['X']
#             X_minus = np.transpose([last_X[0][0], last_X[1][1], last_X[2][2], last_X[3][3]])
#             P_minus = updated[len(updated) - 1]['P']
#
#
#         predict = self._predict(X_minus, P_minus, F, Q, B, std_devs, acc_east, acc_north)
#         predicted.append(predict)
#         last_X = predicted[len(predicted) - 1]['X']
#         if(counter == 1):
#             X_minus = predicted[len(predicted) - 1]['X']
#             counter = counter + 1
#         else:
#             X_minus = np.transpose(last_X)
#         P_minus = predicted[len(predicted) - 1]['P']
#
#         update = self._update(X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat)
#         updated.append(update)
#
#         lng_to_plot.append(update['X'][0][0])
#         lat_to_plot.append(update['X'][1][1])
#
#     """
#     print('Length of updated states')
#     #print(kalmaned_coordinates)
#     print(len(lng_to_plot))
#     print(len(lat_to_plot))
#     print(lng_to_plot[0::100])
#     print(lat_to_plot[0::100])
#     """
#
#     plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'r--')
#     plt.show()
#     return updated
#
#     """
#     diff_lat, diff_lng = [], []
#
#     for m_lng, k_lng in zip(measurement_lng, x_to_plot):
#         diff = m_lng-k_lng
#         diff_lng.append(diff)
#     for m_lat, k_lat in zip(measurement_lat, y_to_plot):
#         diff = m_lat-k_lat
#         diff_lat.append(diff)
#
#
#     print(diff_lng[0::100])
#     print(diff_lat[0::100])
#
#
#     out = '/home/acer/Desktop/szakdoga/code/code_oop/teszt/roszke-szeged/kalmaned_coordinates.csv'
#     with open(out, 'w') as out:
#         for i in kalmaned_coordinates:
#             out.write(str(i[0]) + ',' + str(i[1]) + '\n')
#     """

def pass_kalmaned_list(self, updated, gps):
    # EZ SZAR
    """
    468 kalmaned timestamps list
    468 gps measurement dictinary elements
    """
    timestamps = sorted(list(gps.keys()))
    # print(timestamps)
    kalmaned_dict = {}
    kalmaned_list = []
    for t, k in zip(timestamps, updated):
        measurement = gps[t]
        kalmaned_measurement = k['IM']
        lng_kalmaned = kalmaned_measurement[0]
        lat_kalmaned = kalmaned_measurement[1]
        kalmaned_list.append([measurement['time'], lng_kalmaned, lat_kalmaned])
        kalmaned_dict[t] = {
            'timestamp': measurement['time'],
            'lat': measurement['lat'],
            'lng': measurement['lng'],
            'kalmaned_lng': lng_kalmaned,
            'kalmaned_lat': lat_kalmaned
        }
    return kalmaned_list


def pass_likelihood(self):
    pass


"""
        for i in sorted_t:
            measurement = fused[i]
            
            if (len(measurement) == 8): 
                R[0,0] = measurement['hdop'] * measurement['hdop'] 
                R[1,1] = measurement['hdop'] * measurement['hdop'] 
                R[2,2] = measurement['hdop'] 
                R[3,3] = measurement['hdop'] 
                X = np.transpose([measurement['lng'], measurement['lat'], measurement['vlng'], measurement['vlat']])
                Y = H*X + R

                if not evaluator:
                    X_minus = X
                elif evaluator[len(evaluator) -1] == 1:
                    last_X = each_step[len(each_step)-1]['X']
                    X_minus = last_X[0][0], last_X[1][1], last_X[2][2], last_X[3][3]
                    print(last_X)
                    print('elif 1')
                    
                elif evaluator[len(evaluator) -1] == 0:
                    print(len(evaluator))
                    X_minus = each_step[len(each_step)-1]['X'] 
                    print(X_minus)
                    print('elif 222222222222222222222')
                    
                else:
                    print('Error')
                    return

                
                
                
                
                    
                kalman = Kalman()
                update = kalman.kf_update(X_minus, P, Y, H, R)

                kalmaned.append(update)
                evaluator.append(1)
                each_step.append(update)

                counter = counter + 1
                
            elif (len(measurement) == 4):
                
                Q[0,0] = std_devs['std_dev_acc_east'] * dt* dt
                Q[1,1] = std_devs['std_dev_acc_north'] * dt* dt
                Q[2,2] = std_devs['std_dev_acc_east'] * dt
                Q[3,3] = std_devs['std_dev_acc_north'] * dt
                Q[0,2] = std_devs['std_dev_ln'] * std_devs['std_dev_la']
                Q[1,3] = std_devs['std_dev_vln'] * std_devs['std_dev_vla']
                
                U = np.transpose([measurement['acc_east'], measurement['acc_north']])

                kalman = Kalman()
                predict = kalman.kf_predict(X, P, F, Q, B, U)
                predicted.append(predict)
                evaluator.append(0)
                each_step.append(predict) 
                counter = counter + 1
            else:
                print('kalman filter can\'t apply your shit man')
"""

"""
https://github.com/akshaychawla/1D-Kalman-Filter
https://dsp.stackexchange.com/questions/8860/kalman-filter-for-position-and-velocity-introducing-speed-estimates
https://dsp.stackexchange.com/questions/38045/even-more-on-kalman-filter-for-position-and-velocity?noredirect=1&lq=1
https://dsp.stackexchange.com/questions/48343/python-how-can-i-improve-my-1d-kalman-filter-estimate
https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python
https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
https://www.reddit.com/r/computervision/comments/35y4kj/looking_for_a_python_example_of_a_simple_2d/
https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1 !!!!!!!!!!!!!!!!
https://balzer82.github.io/Kalman/
https://towardsdatascience.com/kalman-filter-an-algorithm-for-making-sense-from-the-insights-of-various-sensors-fused-together-ddf67597f35e
https://gist.github.com/manicai/922976
"""