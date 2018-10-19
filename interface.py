from gps_data import GPS_data
from imu_data import IMU_data
from kalman import Kalman
from initial_params import Initial_params
import numpy as np
import matplotlib.pyplot as plt

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
        for t in timestamps:
            values = gps[t]
            ln.append(values['lng'])
            la.append(values['lat'])
            vln.append(values['vlng'])
            vla.append(values['vlat'])
            hdop.append(values['hdop'])
            time.append(values['time'])
        return {'ln': ln, 'la': la, 'vln': vln, 'vla': vla, 'hdop': hdop, 'time': time}
    
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
        
    def _pass_std_devs(self, gps, acc):
        gps_lists = self.pass_gps_list(gps)
        acc_lists = self.pass_acc_list(acc)
        std_dev_ln = np.std(gps_lists['ln'])
        std_dev_la = np.std(gps_lists['la'])
        std_dev_vln = np.std(gps_lists['vln'])
        std_dev_vla = np.std(gps_lists['vla'])
        std_dev_acc_east = np.std(acc_lists['acc_east'])
        std_dev_acc_north = np.std(acc_lists['acc_north'])

        return {'std_dev_ln': std_dev_ln,'std_dev_la': std_dev_la, 'std_dev_vln': std_dev_vln,'std_dev_vla': std_dev_vla, 'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}

    def _update(gps, acc, X, P, Y, H, R):
        pass
    def _predict(gps, acc, X, P, F, Q, B, U):
        pass

    def interpolate_gps_data(self, acc, gps):

        gps_lists = self.pass_gps_list(gps)
        acc_lists = self.pass_acc_list(acc)
        
        acc_time = acc_lists['acc_time']
        print(type(acc_time))
        acc_east = acc_lists['acc_east']
        acc_north = acc_lists['acc_north']
        acc_down = acc_lists['acc_down']
        
        gps_time = gps_lists['time']
        gps_lng = gps_lists['ln']
        gps_lat = gps_lists['la']
        gps_vln = gps_lists['vln']
        gps_vla = gps_lists['vla']
        gps_hdop = gps_lists['hdop']
        

        interpolated_lng = np.interp(acc_time, gps_time, gps_lng)
        interpolated_lat = np.interp(acc_time, gps_time, gps_lat)
        interpolated_vlng = np.interp(acc_time, gps_time, gps_vln)
        interpolated_vlat = np.interp(acc_time, gps_time, gps_vla)
        interpolated_hdop = np.interp(acc_time, gps_time, gps_hdop)
        if(len(acc_time) == len(acc_down) == len(acc_east) == len(acc_north) == len(interpolated_lat) == len(interpolated_lng) == len(interpolated_vlat) == len(interpolated_vlng) == len(interpolated_hdop)):
            print('EQUAL')
        else:
            print('NEM')

        return {'acc_time': acc_time, 'lng': interpolated_lng, 'lat': interpolated_lat, 'vlng': interpolated_vlng, 'vlat': interpolated_vlat, 'hdop': interpolated_hdop, 'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down}



        """
        for pos in zip(acc_time, interpolated_lng, interpolated_lat, acc_east, acc_north, acc_down):
            interpolated_attribute_table.append(pos)

        self._write_attr_table_to_file(interpolated_attribute_table)
        return interpolated_attribute_table
        """

    def get_kalmaned_coordinates(self, gps, acc):
        kalman = Kalman()
        gps_lists = self.pass_gps_list(gps)
        acc_lists = self.pass_acc_list(acc)

        interpolated_attribute_table = self.interpolate_gps_data(acc, gps)

        init = Initial_params()
        init_params = init.get_initial_parameters()
        P = init_params['P']
        H = init_params['H']
        R = init_params['R']
        Q = init_params['Q']
        F = init_params['F']
        B = init_params['B']
        dt = 1

        std_devs = self._pass_std_devs(gps, acc)

        fused = {**acc, **gps}
        sorted_timestamps_of_all_measurements = sorted(list(fused.keys()))
        
        updated = []
        predicted = []
        last_step = []
        each_state = []
        X = 0
        X_minus = 0
        counter = 0
        for t in sorted_timestamps_of_all_measurements:
            #THIS HAS TO BE CHECKED!
            measurement = fused[t]
            #print(measurement)
            
            #if we got a measurement from GPS
            if (len(measurement) == 8): 
                if len(last_step) != 0: 
                    last_step_was = last_step[len(last_step)-1]
                
                R[0,0] = measurement['hdop'] * measurement['hdop'] 
                R[1,1] = measurement['hdop'] * measurement['hdop'] 
                R[2,2] = measurement['hdop'] 
                R[3,3] = measurement['hdop'] 
                
                # so predict will be the first, but a state has to be initialized
                if len(predicted) == 0:
                    X = np.transpose([measurement['lng'], measurement['lat'], measurement['vlng'], measurement['vlat']])
                    updated.append({'X':X,'P':P})
                    each_state.append(X)
                    last_step.append('updated')
                    continue
                #if last step was a predcition, use its state as the state input, and this measurement for Y
                elif(last_step_was == 'predicted'):
                    
                    X = np.transpose([measurement['lng'], measurement['lat'], measurement['vlng'], measurement['vlat']])
                    Y = H*X + R
                    P_minus = predicted[len(predicted) - 1]['P']
                    X_minus = predicted[len(predicted) - 1]['X']
                    if (len(X_minus) > 4):
                        print(X_minus)
                        break
                    update = kalman.kf_update(X_minus, P_minus, Y, H, R)

                    update['X'] = np.transpose([update['X'][0][0], update['X'][1][1], update['X'][2][2], update['X'][3][3]])
                    updated.append(update)
                    last_step.append('updated')
                    #if last step was an update, use its state as the state input, and this measurement for Y
                elif(last_step_was == 'updated'):

                    X = np.transpose([measurement['lng'], measurement['lat'], measurement['vlng'], measurement['vlat']])
                    Y = H*X + R
                    P_minus = updated[len(updated) - 1]['P']
                    X_minus = np.transpose(updated[len(updated) - 1]['X'])
                    if (len(X_minus) > 4):
                        print(X_minus)
                        break
                    update = kalman.kf_update(X_minus, P, Y, H, R)
                    update['X'] = [update['X'][0][0], update['X'][1][1], update['X'][2][2], update['X'][3][3]]
                    updated.append(update)
                    last_step.append('updated')

            #if we got a control vector
            elif (len(measurement) == 4):
                if len(last_step) != 0:
                    last_step_was = last_step[len(last_step)-1]

                Q[0,0] = std_devs['std_dev_acc_east'] * dt* dt
                Q[1,1] = std_devs['std_dev_acc_north'] * dt* dt 
                Q[2,2] = std_devs['std_dev_acc_east'] * dt
                Q[3,3] = std_devs['std_dev_acc_north'] * dt
                Q[0,2] = std_devs['std_dev_ln'] * std_devs['std_dev_la']
                Q[1,3] = std_devs['std_dev_vln'] * std_devs['std_dev_vla']
                
                U = np.transpose([measurement['acc_east'], measurement['acc_north']])
                if len(predicted) == 0:
                    #if updated is empty, do it first
                    if len(updated) == 0:
                        continue
                    #otherwise use last state from measurement
                    else:
                        X_minus = updated[len(updated) - 1]['X']
                        P_minus = updated[len(updated) - 1]['P']
                        predict = kalman.kf_predict(X_minus, P, F, Q, B, U)
                        predicted.append(predict)
                        last_step.append('predicted')
                #if last step was prediction, use the parameters from the last predicted state
                elif (last_step_was == 'predicted'): #good
                    P_minus = predicted[len(predicted) - 1]['P']
                    X_minus = predicted[len(predicted) - 1]['X']
                    if (len(X_minus) > 4):
                        print(X_minus)
                        break
                    predict = kalman.kf_predict(X_minus, P, F, Q, B, U)
                    predicted.append(predict)
                    last_step.append('predicted')
                #if last step was update, use the state from the last updated state
                elif (last_step_was == 'updated'): 
                    P_minus = updated[len(updated) - 1]['P']
                    X_minus = np.transpose(updated[len(updated) - 1]['X']) 
                    if (len(X_minus) > 4):
                        print(X_minus)
                        break
                    predict = kalman.kf_predict(X_minus, P, F, Q, B, U)
                    predicted.append(predict)
                    last_step.append('predicted')
            else:
                print('kalman filter can\'t apply your shit man')
                break
        kalmaned_coordinates = [i['X'] for i in updated]
        print('Length of updated states')
        #print(kalmaned_coordinates)
        print(len(last_step))
        print(len(kalmaned_coordinates))
        print(len(kalmaned_coordinates[0]))
        print(kalmaned_coordinates[0::100])

        x_to_plot = []
        y_to_plot = []
        for i in kalmaned_coordinates:
            #print(i)
            x_to_plot.append(i[0])
            y_to_plot.append(i[1])

        plt.plot(gps_lists['ln'], gps_lists['la'], 'bs', x_to_plot, y_to_plot, 'r--')
        plt.show()
        return updated

        """
        x_to_plot = []
        y_to_plot = []
        for i in kalmaned_coordinates:
            #print(i)
            x_to_plot.append(i[0][0])
            y_to_plot.append(i[1][1])
        

        print(type(gps_lists['ln']))
        print(len(gps_lists['ln']))
        print(type(gps_lists['la']))
        print(len(gps_lists['la']))
        print(len(x_to_plot))
        print(len(y_to_plot))

    
        
        plt.plot(gps_lists['ln'], gps_lists['la'], 'bs', x_to_plot, y_to_plot, 'r--')
        plt.show()
        print('ide is eljut')
        """
        """
        diff_lat, diff_lng = [], []

        for m_lng, k_lng in zip(measurement_lng, x_to_plot):
            diff = m_lng-k_lng
            diff_lng.append(diff)
        for m_lat, k_lat in zip(measurement_lat, y_to_plot):
            diff = m_lat-k_lat
            diff_lat.append(diff)
        

        print(diff_lng[0::100])
        print(diff_lat[0::100])
        

        out = '/home/acer/Desktop/szakdoga/code/code_oop/teszt/roszke-szeged/kalmaned_coordinates.csv'
        with open(out, 'w') as out:
            for i in kalmaned_coordinates:
                out.write(str(i[0]) + ',' + str(i[1]) + '\n')
        """
    
    def pass_kalmaned_list(self, updated, gps):
        #EZ SZAR
        """
        468 kalmaned timestamps list
        468 gps measurement dictinary elements
        """
        timestamps = sorted(list(gps.keys()))
        #print(timestamps)
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

        
