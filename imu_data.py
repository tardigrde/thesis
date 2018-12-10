from madgwick.madgwickahrs import MadgwickAHRS
from pyquaternion import Quaternion
import pandas as pd
import numpy as np
import math

class IMU_data:

    # Input: filename
    # Output: Dictionary of accelearation values.

    def __init__(self, filename):
        #with open(filename, 'rt') as inputfile:
        #    self.file_path = inputfile
        self.filename = filename
        self.measurement_dictionary = {}
        self.quaternion_list = []
        self.rotation_matrix_list = []
        self.absolute_acceleration_list = []
        self.true_absolute_acceleration_dictionary = {}
        self.acceleration_std = {}

        
        
    def trim_datatable(self):
        """
        Trim the huge datatable, which is passed to the class using Pandas dataframe.
        Only the below stated columns remain.

        Using Pandas dataframe for the sake of simplicity.
        """
            
        raw_data = pd.read_csv(self.filename, sep=',')
        df_raw = pd.DataFrame(raw_data)
        self.trimmed_dataframe = df_raw[['Timestamp', 'accelX', 'accelY', 'accelZ', 
        'gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)', 'calMagX', 'calMagY', 'calMagZ']]
        self.trimmed_dataframe.head()
        
        return self.trimmed_dataframe    
        
    
    def extract_columns(self, trimmed_dataframe):
        """
        Extract columns from the passed dataframe.
        """

        time = [str(t) for t in trimmed_dataframe.iloc[:,0]]
        acc_x = [aX for aX in trimmed_dataframe.iloc[:,1]]
        acc_y = [aY for aY in trimmed_dataframe.iloc[:,2]]
        acc_z = [aZ for aZ in trimmed_dataframe.iloc[:,3]]
        gyro_x = [gX for gX in trimmed_dataframe.iloc[:,4]]
        gyro_y = [gY for gY in trimmed_dataframe.iloc[:,5]]
        gyro_z = [gZ for gZ in trimmed_dataframe.iloc[:,6]]
        mag_x = [gX for gX in trimmed_dataframe.iloc[:,7]]
        mag_y = [gY for gY in trimmed_dataframe.iloc[:,8]]
        mag_z = [gZ for gZ in trimmed_dataframe.iloc[:,9]]

        def _milisecondify(t):
            t = (str(t).split(' '))[1].replace(':', '.').split('.')
            ms = int(t[3]) + (int(t[2])* 1000) + (int(t[1])*60*1000) + (int(t[0])*60*60*1000)
            return ms 
        
        self.timestamp = [_milisecondify(t) for t in time]

        
        self.measurement_dictionary = {
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z, 
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z, 
            'mag_x': mag_x, 'mag_y': mag_y, 'mag_z': mag_z}
        
        return self.measurement_dictionary
    
    
    #converting sensor reading into quaternions (see:  )
    def get_quaternion(self, measurement_dictionary):  

        """
        Converting raw accelerometer, gyroscope and magnetometer readings piece by piece to quaternions.
        Input: dictionary of acceleroeter, gyroscope and magnetometer measurements 
        (as specified in extract_columns method.)
        I use Morgil's implementation of Madgwick Quaternion Update. 
        Source: https://github.com/morgil/madgwick_py
        More on quaternions: http://mathworld.wolfram.com/Quaternion.html
        """  
        
        no_of_measurments = len(self.timestamp)
        md = self.measurement_dictionary
        self.quaternion_list = []
  
        for i in range(no_of_measurments):
            
            acc = [md['acc_x'][i], md['acc_y'][i], md['acc_z'][i]]
            gyro = [md['gyro_x'][i], md['gyro_y'][i], md['gyro_z'][i]]
            mag = [md['mag_x'][i], md['mag_y'][i], md['mag_z'][i]]

            # the Madgwick algorythm is unable to run if there is zero in your readings (even if it's false reading)
            a = acc
            a[a == 0] = 0.1
            g = gyro
            g[g == 0] = 0.1
            m = mag
            m[m == 0] = 0.1

            def _convert(acc, gyro, mag):
                MadgwickAHRS.update(MadgwickAHRS, gyro, acc, mag)
                Q = MadgwickAHRS.quaternion
                quaternion = Quaternion([float(Q[0]), float(Q[1]), float(Q[2]), float(Q[3])])
                return quaternion

            self.quaternion_list.append(_convert(acc, gyro, mag))
        return self.quaternion_list
        

        # converts quaternions to rotation matrices
    def get_rotation_matrix (self, quaternion_list):

        """
        Converting the above quaternions (specified in get_quaternions method) into rotation matrices.
        I use kieranwynn's pyquaternion library to do so (http://kieranwynn.github.io/pyquaternion/).
        Input: list of quaternions.
        Output: list of rotation matrices.
        More on rotation matrices: http://mathworld.wolfram.com/RotationMatrix.html.
        """
        
        no_of_measurments = len(self.timestamp)
        rotation_matrix = []
        
        for i in range(no_of_measurments):
            rotm = self.quaternion_list[i].rotation_matrix.tolist()
            rotation_matrix.append(rotm)
        
        self.rotation_matrix_list = rotation_matrix
        return self.rotation_matrix_list        
    
    def get_absolute_acc(self, rotation_matrix_list):
        """
        Calculates absolute acceleration from the rotation matrix and the measured acceleration vector.
        Absolute acceleration means East-North-Up in this case.
        Input: list of rotation matrices, acceleration readings.
        Output: list of absolute accelerations.
        """

        md = self.measurement_dictionary
        no_of_measurments = len(self.timestamp)
       
        for i in range(no_of_measurments):

            #GRAVITY FACTOR HAS TO BE ACCOUNTED FOR!!!!!!!
            #ned_acc[2] += 1 CHANGE if MOVEMENT is FREE
            
            ned_acc = np.dot(np.linalg.inv(rotation_matrix_list[i]), 
            np.transpose([float(md['acc_x'][i]), float(md['acc_y'][i]), float(md['acc_z'][i])]))
            
            self.absolute_acceleration_list.append(ned_acc.tolist())

        return self.absolute_acceleration_list
    
    def get_true_abs_acc_dict(self, absolute_acceleration_list):
        """
        This function rotataes the "absolute" acceleration frame so north means true north, 
        instead of magnetic north. For this, I use Scott Lobdell's modified code,
        see: http://scottlobdell.me/2017/01/gps-accelerometer-sensor-fusion-kalman-filter-practical-walkthrough/.
        For now magnetic declination offset is manually set.
        Input: list of absolute accelerations.
        Output: dictionary of absolute acceleration with true north.
        """

        accelerations = self.absolute_acceleration_list
        timestamp = self.timestamp
        no_of_measurements = len(timestamp)

        """
        test if timestamps and accelerations have the same length
        """
        
            
        def _calculate_true_acceleration(i):
            
            row = accelerations[i]
            magnetic_declination_offset = 4.96666667 # CSONGRÁD MEGYE

            sin_magnetic_declination = math.sin(math.radians(magnetic_declination_offset))
            cos_magnetic_declination = math.cos(math.radians(magnetic_declination_offset))

            eastern_north_component = sin_magnetic_declination * row[0] # row[0] is acceleration east
            northern_east_component = -sin_magnetic_declination * row[1] # row[1] is acceleration north

            northern_north_component = cos_magnetic_declination * row[1]
            eastern_east_component = cos_magnetic_declination * row[0]

            acc_east = eastern_east_component + eastern_north_component
            acc_north = northern_north_component + northern_east_component

            #self.acceleration_list.append([timestamp[i], acc_east, acc_north, row[2]])
                
            
            return {'time': timestamp[i],'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': row[2]}

        for i in range(no_of_measurements):
            true_acceleration = _calculate_true_acceleration(i)
            self.true_absolute_acceleration_dictionary[timestamp[i]] = true_acceleration
            
        
        
        return self.true_absolute_acceleration_dictionary
