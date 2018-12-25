"""Spreadsheet Reader

"""
import pandas as pd


def _read_datatable():
    """
    Read the huge datatable, which is passed to the function using Pandas dataframe.

    Only the below stated columns remain.

    Using Pandas dataframe for the sake of simplicity.
    """
    print('hello world')

#     raw_data = pd.read_csv(filename, sep=',')
#     dataframe = pd.DataFrame(raw_data)
#     return dataframe

read_datatable()




def _trim_datatable(dataframe):
    trimmed_dataframe = dataframe[
        ['Timestamp', 'accelX', 'accelY', 'accelZ', 'gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)', 'calMagX',
         'calMagY', 'calMagZ']
    ]
    return trimmed_dataframe.head()


def _milisecondify(t):
    """

    :param t: Human readable timestamp.
    :return: Unix timestamp.
    """
    t = (str(t).split(' '))[1].replace(':', '.').split('.')
    ms = int(t[3]) + (int(t[2]) * 1000) + (int(t[1]) * 60 * 1000) + (int(t[0]) * 60 * 60 * 1000)
    return ms

