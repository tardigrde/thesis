# import pytest
# from .. import imu_data_reader
#
#
# """
#     Testing imu_data.py.
#     imu_data has to have  forms like:
#         quaternions
#         rotation matrix
#         absolute acceleration to true north
# """
#
#
# def test_read_datatable(filename):
#     """
#     This script requires that `pandas` be installed within the Python
#     environment you are running this file in.
#     :return:
#     """
#
#     # read csv into dataframe
#     dataframe = imu_data_reader.read_datatable(filename)
#     # should have no of columns
#     # should have no of rows
#     assert sum([1, 2, 3]), 6, "Should be 6")
#
#
# # def test_trim_datatable(self):
# #     self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")
#
#
# if __name__ == '__main__':
#     unittest.main()
