# def trim_datatable(self):
#     """
#     Trim the huge datatable, which is passed to the class using Pandas dataframe.
#     Only the below stated columns remain.
#
#     Using Pandas dataframe for the sake of simplicity.
#     """
#
#     raw_data = pd.read_csv(self.filename, sep=',')
#     df_raw = pd.DataFrame(raw_data)
#     """
#     self.trimmed_dataframe = df_raw[['Timestamp', 'accelX', 'accelY', 'accelZ',
#     'gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)', 'calMagX', 'calMagY', 'calMagZ']]
#     self.trimmed_dataframe.head()
#     """
#
#     measurment = []
#     for index, row in df_raw.iterrows():
#         measurment = ['Timestamp': row['Timestamp'], 'accelX': row['accelX'], 'accelY': row['accelY'], 'accelZ': row['accelZ'], 'gyroX': row['gyroX'], 'gyroY': row['gyroY'], 'gyroZ': row['gyroZ'], 'calMagX': row['calMagX'], 'calMagY': row['calMagY'], 'calMagZ': row['calMagZ']]
#
#
#
#     return self.trimmed_dataframe