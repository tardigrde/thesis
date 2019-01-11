
# def _predict(X_minus, P_minus, F, Q, B, std_devs, acc_east, acc_north):
#     """
#
#     :param X_minus:
#     :param P_minus:
#     :param F:
#     :param Q:
#     :param B:
#     :param std_devs:
#     :param acc_east:
#     :param acc_north:
#     :return:
#     """
#     kalman = Kalman()
#     dt = 1
#     Q[0, 0] = (std_devs['std_dev_acc_east'] * dt * dt / 2) ** 2
#     Q[1, 1] = (std_devs['std_dev_acc_north'] * dt * dt / 2) ** 2
#     Q[2, 2] = (std_devs['std_dev_acc_east'] * dt) ** 2
#     Q[3, 3] = (std_devs['std_dev_acc_north'] * dt) ** 2
#     Q[0, 2] = (std_devs['std_dev_acc_east'] * dt * dt / 2) * (std_devs['std_dev_acc_east'] * dt)
#     Q[1, 3] = (std_devs['std_dev_acc_north'] * dt * dt / 2) * (std_devs['std_dev_acc_north'] * dt)
#
#     U = np.transpose([acc_east, acc_north])
#
#     # print('Q: {} and U: {}'.format(Q, U))
#
#     predict = kalman.kf_predict(X_minus, P_minus, F, Q, B, U)
#     return predict
#
#
# def _update(X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat):
#     """
#
#     :param X_minus:
#     :param P_minus:
#     :param H:
#     :param R:
#     :param hdop:
#     :param lng:
#     :param lat:
#     :param vlng:
#     :param vlat:
#     :return:
#     """
#     kalman = Kalman()
#     # print("vl: {}\n va: {}".format(vlng, vlat))
#     R[0][0] = hdop * hdop
#     R[1][1] = hdop * hdop
#     R[2][2] = hdop * hdop  # *10 works better
#     R[3][3] = hdop * hdop
#     # print('R is {}'.format(R))
#     X = np.transpose([lng, lat, vlng, vlat])
#     # print('X is {}'.format(X))
#     Y = X * H
#     # print('Y is {}'.format(Y))
#
#     update = kalman.kf_update(X_minus, P_minus, Y, H, R)
#
#     return update

# print('Length of updated states {}'.format(len(updated)))
# #print('Predicted STATE is {}'.format(predict['X']))
# plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'r--')
# plt.show()
#
# out = './teszt/szeged_trolli_teszt/kalmaned_coordinates21.csv'
# with open(out, 'w') as out:
#     for ln, la in zip(lng_to_plot, lat_to_plot):
#         out.write(str(ln) + ',' + str(la) + '\n')
#
# out_og = './teszt/szeged_trolli_teszt/og_coords.csv'
# with open(out_og, 'w') as out:
#     for ln, la in zip(og_lng, og_lat):
#         out.write(str(ln) + ',' + str(la) + '\n')
# return updated

# diff_lat, diff_lng = [], []
# for m_lng, k_lng in zip(measurement_lng, x_to_plot):
#     diff = m_lng - k_lng
#     diff_lng.append(diff)
# for m_lat, k_lat in zip(measurement_lat, y_to_plot):
#     diff = m_lat - k_lat
#     diff_lat.append(diff)
#
# print(diff_lng[0::100])
# print(diff_lat[0::100])
#
# out = '/home/acer/Desktop/szakdoga/code/code_oop/teszt/roszke-szeged/kalmaned_coordinates.csv'
# with open(out, 'w') as out:
#     for i in kalmaned_coordinates:
#         out.write(str(i[0]) + ',' + str(i[1]) + '\n')
# gps_lists = pass_gps_list(gps)
# acc_lists = pass_acc_list(acc)
#
# acc_time = acc_lists['acc_time']
# acc_east = acc_lists['acc_east']
# acc_north = acc_lists['acc_north']
# acc_down = acc_lists['acc_down']
# print("at: {}, ae: {}, an: {}, ad: {}".format(len(acc_time), len(acc_east), len(acc_north), len(acc_down)))
#
# gps_time = gps_lists['time']
# gps_lng = gps_lists['ln']
# gps_lat = gps_lists['la']
# gps_vln = gps_lists['vln']
# gps_vla = gps_lists['vla']
# gps_hdop = gps_lists['hdop']
# print("gt: {}, gln: {}, glt: {}, gvln: {}, gvlt: {}, hdop: {}".format(len(gps_time), len(gps_lng), len(gps_lat), len(gps_vln), len(gps_vla), len(gps_hdop)))


# def do_pykalmaning(self, acc, gps):
#     gps_lists = self.pass_gps_list(gps)
#     acc_lists = self.pass_acc_list(acc)
#
#     acc_time = acc_lists['acc_time']
#     acc_east = acc_lists['acc_east']
#     acc_north = acc_lists['acc_north']
#     acc_down = acc_lists['acc_down']
#
#     gps_time = gps_lists['time']
#     gps_lng = gps_lists['ln']
#     gps_lat = gps_lists['la']
#     gps_vlng = gps_lists['vln']
#     gps_vlat = gps_lists['vla']
#     gps_hdop = gps_lists['hdop']
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
#     # interpolated_attribute_table = self.interpolate_gps_data(acc, gps)
#     # acc_time = interpolated_attribute_table['acc_time']
#     # acc_east = interpolated_attribute_table['acc_east']
#     # acc_north = interpolated_attribute_table['acc_north']
#     # gps_lng = interpolated_attribute_table['lng']
#     # gps_lat = interpolated_attribute_table['lat']
#     # gps_vlng = interpolated_attribute_table['vlng']
#     # gps_vlat = interpolated_attribute_table['vlat']
#     # gps_hdop = interpolated_attribute_table['hdop']
#
#     std_devs = self._pass_std_devs(acc_east, acc_north)
#
#     updated = []
#     predicted = []
#     lng_to_plot = []
#     lat_to_plot = []
#     og_lng, og_lat = [], []
#
#     counter = 0
#     for acc_time, gps_time, a_east, a_north, lat, lng, vlat, vlng, hdop in zip(acc_time, gps_time, acc_east, acc_north, gps_lat, gps_lng, gps_vlat, gps_vlng, gps_hdop):
#
#         og_lng.append(lng)
#         og_lat.append(lat)
#
#         if counter == 0:
#             X_minus = np.transpose([lng, lat, vlng, vlat])
#             P_minus = P
#             counter = counter + 1
#         else:
#             X_minus = updated[len(updated) - 1]['X']
#             P_minus = updated[len(updated) - 1]['P']
#
#         predict = self._predict(X_minus, P_minus, F, Q, B, std_devs, a_east, a_north) # 1
#         #print('Predicted STATE is {}'.format(predict['X']))
#         predicted.append(predict)
#
#         X_minus = predicted[len(predicted) - 1]['X']
#         P_minus = predicted[len(predicted) - 1]['P'] #2,3
#
#         update = self._update(X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat)
#         updated.append(update)
#
#         lng_to_plot.append(update['X'][0][0]) ## CHECK THIS
#         lat_to_plot.append(update['X'][1][1])
#
#     plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'r--')
#     plt.show()
#     #print()
#     out = './teszt/szeged_trolli_teszt/kalmaned_coordinates15.csv'
#     with open(out, 'w') as out:
#         for ln, la in zip(lng_to_plot, lat_to_plot):
#             out.write(str(ln) + ',' + str(la) + '\n')
#
#     # out_og = './teszt/szeged_trolli_teszt/og_coords.csv'
#     # with open(out_og, 'w') as out:
#     #     for ln, la in zip(og_lng, og_lat):
#     #         out.write(str(ln) + ',' + str(la) + '\n')

# def pass_kalmaned_list(self, updated, gps):
#     # EZ SZAR
#     """
#     468 kalmaned timestamps list
#     468 gps measurement dictinary elements
#     """
#     timestamps = sorted(list(gps.keys()))
#     # print(timestamps)
#     kalmaned_dict = {}
#     kalmaned_list = []
#     for t, k in zip(timestamps, updated):
#         measurement = gps[t]
#         kalmaned_measurement = k['IM']
#         lng_kalmaned = kalmaned_measurement[0]
#         lat_kalmaned = kalmaned_measurement[1]
#         kalmaned_list.append([measurement['time'], lng_kalmaned, lat_kalmaned])
#         kalmaned_dict[t] = {
#             'timestamp': measurement['time'],
#             'lat': measurement['lat'],
#             'lng': measurement['lng'],
#             'kalmaned_lng': lng_kalmaned,
#             'kalmaned_lat': lat_kalmaned
#         }
#     return kalmaned_list
#
#
# def pass_likelihood(self):
#     pass

# in_proj = Proj(init='epsg:4326')
# out_proj = Proj(init='epsg:23700')
# lng_eov, lat_eov = transform(in_proj, out_proj, lng, lat)
# print(measurement)

# def _plot_ghashed_against_real_coords(measurement, i):
#     lng = measurement['lng']
#     og_lng.append(lng)
#     lat = measurement['lat']
#     og_lat.append(lat)
#     lng_to_add, lat_to_add = Geohash.decode(measurement['geohashed'])
#
#     if (not lng_to_plot and not lat_to_plot):
#         lng_to_plot.append(lng_to_add)
#         lat_to_plot.append(lat_to_add)
#         i = i + 1
#     elif (lng_to_add == lng_to_plot[i] and lat_to_add == lat_to_plot[i]):
#         pass
#     else:
#         lng_to_plot.append(lng_to_add)
#         lat_to_plot.append(lat_to_add)
#         i = i + 1
#         # to_plot.append([lng_to_add, lat_to_plot])
#
# if (len(measurement) == 9):
#     _plot_ghashed_against_real_coords(measurement, i)

# updated = []
# predicted = []
# last_step_was = 'none'
# last_step = []
# lng_to_plot = []
# lat_to_plot = []
# og_lng, og_lat = [], []
#
# for t in sorted_timestamps_of_all_measurements:
#     # THIS HAS TO BE CHECKED!
#     measurement = fused[t]
#
#     # if we got a measurement from GPS
#     if len(measurement) == 6:
#         lng = measurement['lng']
#         og_lng.append(lng)
#         lat = measurement['lat']
#         og_lat.append(lat)
#         vlng = measurement['vlng']
#         vlat = measurement['vlat']
#         hdop = measurement['hdop']
#
#         # so predict will be the first, but a state has to be initialized
#         if last_step_was == 'none':
#             X = np.transpose([measurement['lng'], measurement['lat'], measurement['vlng'], measurement['vlat']])
#             updated.append({'X': X, 'P': P})
#             last_step_was = 'update'
#             last_step.append('update')
#             lng_to_plot.append(X[0])
#             lat_to_plot.append(X[1])
#
#         # if last step was a predcition, use its state as the state input, and this measurement for Y
#         elif last_step_was == 'predict':
#             P_minus = predicted[len(predicted) - 1]['P']
#             X_minus = predicted[len(predicted) - 1]['X']
#             update = _update(X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat)
#             updated.append(update)
#             lng_to_plot.append(update['X'][0][0])
#             lat_to_plot.append(update['X'][1][1])
#             last_step_was = 'update'
#             last_step.append('update')
#
#         # if last step was an update, use its state as the state input, and this measurement for Y
#         elif last_step_was == 'update':
#             P_minus = updated[len(updated) - 1]['P']
#             X_minus = updated[len(updated) - 1]['X']
#             update = _update(X_minus, P_minus, H, R, hdop, lng, lat, vlng, vlat)
#             lng_to_plot.append(update['X'][0][0])
#             lat_to_plot.append(update['X'][1][1])
#             updated.append(update)
#             last_step_was = 'update'
#             last_step.append('update')
#
#     elif len(measurement) == 4:
#         acc_east = measurement['acc_east']
#         acc_north = measurement['acc_north']
#
#         if last_step_was == 'predict':
#             X_minus = predicted[len(predicted) - 1]['X']
#             P_minus = predicted[len(predicted) - 1]['P']
#             predict = _predict(X_minus, P_minus, F, Q, B, std_devs, acc_east, acc_north)
#             predicted.append(predict)
#             last_step_was = 'predict'
#             last_step.append('predict')
#         elif last_step_was == 'update':
#             X_minus = updated[len(updated) - 1]['X']
#             P_minus = updated[len(updated) - 1]['P']
#             predict = _predict(X_minus, P_minus, F, Q, B, std_devs, acc_east, acc_north)
#             predicted.append(predict)
#             last_step_was = 'predict'
#             last_step.append('predict')
#     else:
#         print('The length of measurement is not one of 4 or 6.')
#         break
#
# # print('Length of lng is: {}'.format(len(lng_to_plot)))
# # print('Length of lat is: {}'.format(len(lat_to_plot)))
# # print('Length of og_lng is: {}'.format(len(og_lng)))
# # print('Length of og_lat is: {}'.format(len(og_lat)))
# print('Length of steps is: {}'.format(len(last_step)))
#
# plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'ro')
# plt.show()
#
# out = './teszt/szeged_trolli_teszt/nointerpolation/kalmaned.csv'
# with open(out, 'w') as out:
#     for ln, la in zip(lng_to_plot, lat_to_plot):
#         out.write(str(ln) + ',' + str(la) + '\n')
# out_og = './teszt/szeged_trolli_teszt/nointerpolation/og_coords.csv'
# with open(out_og, 'w') as out:
#     for ln, la in zip(og_lng, og_lat):
#         out.write(str(ln) + ',' + str(la) + '\n')
#
# return updated
