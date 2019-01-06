from interface import Interface
import numpy as np

class Point:
    def __init__(self):
        pass
        
    """
    def get_kalmaned():
        kalmaned = Segment.pass_kalamaned_coordinates

    def get_acc():
        acc_up_per_down = Segment.pass_z_axis

    kalmaned = get_kalmaned
    acc = get_acc
    """
    def _extract_acc_time(self, acc_data):
        acc_time = [i[0] for i in acc_data]
        return acc_time
    def _extract_acc_east(self, acc_data):
        acc_east = [i[1] for i in acc_data]
        return acc_east
    def _extract_acc_north(self, acc_data):
        acc_north = [i[2] for i in acc_data]
        return acc_north
    def _extract_acc_down(self, acc_data):
        acc_down = [i[3] for i in acc_data]
        return acc_down

    def _extract_kalmaned_time(self, kalmaned):
        kalmaned_time = [i[0] for i in kalmaned]
        return kalmaned_time
    def _extract_kalmaned_lng(self, kalmaned):
        kalmaned_lng = [i[1] for i in kalmaned]
        return kalmaned_lng
    def _extract_kalmaned_lat(self, kalmaned):
        kalmaned_lat = [i[2] for i in kalmaned]
        return kalmaned_lat

    def _write_attr_table_to_file(self, interpolated_attribute_table):
        out = r'/home/acer/Desktop/szakdoga/code/code_oop/teszt/roszke-szeged/kalmaned_with_attributes.csv'

        with open(out, 'w') as out:
            for i in interpolated_attribute_table:
                out.write(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + ',' + str(i[3]) + ',' + str(i[4]) + ',' + str(i[5]) + '\n')


    def interpolate_kalmaned_coordinates(self, acc_data, kalmaned):

        interpolated_attribute_table = []
        
        acc_time = self._extract_acc_time(acc_data)
        acc_east = self._extract_acc_east(acc_data)
        acc_north = self._extract_acc_north(acc_data)
        acc_down = self._extract_acc_down(acc_data)
        
        kalmaned_time = self._extract_kalmaned_time(kalmaned)
        kalmaned_lng = self._extract_kalmaned_lng(kalmaned)
        kalmaned_lat = self._extract_kalmaned_lat(kalmaned)

        interpolated_lng = np.interp(acc_time, kalmaned_time, kalmaned_lng)
        interpolated_lat = np.interp(acc_time, kalmaned_time, kalmaned_lat)

        for pos in zip(acc_time, interpolated_lng, interpolated_lat, acc_east, acc_north, acc_down):
            interpolated_attribute_table.append(pos)

        self._write_attr_table_to_file(interpolated_attribute_table)
        return interpolated_attribute_table

    

    def assign_attributes_to_intepolated_points():
        #straightforward
        #id, timestamp, lat, lng, accx, accy, accz
        return attributed_points

    def segment_dataset():
        #find an algorythm
        #for example bind 10 points together, if the previous was ascending: ; if the next one is ascending: ;
        #if the previous was descending: ; if the next is descending: ;
        # but first ask Gudi !!
        pass
