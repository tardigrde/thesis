from pyproj import Proj, transform
import Geohash
import pynmea2
import math


class GPS_data:
    """
    NMEA GPS sentences class.
    Input: log file.
    Output: gps data output in a dictionary.
    In the input log file every line is a nmea sentece. Every seven sentences make one measurement.

    """

    def __init__(self, filename):
        with open(filename, 'rt') as inputfile:
            self.readed_data = inputfile.readlines()

        self.extracted_data = []
        self.measurement_dictionary = {}
        self.final_lng = []
        self.final_lat = []
        self.i = -1

    def _transform_gsa(self, parsed_sentence):
        hdop = float(parsed_sentence.hdop)
        gsa = {'hdop': hdop}
        return gsa

    def _transform_vtg(self, parsed_sentence):
        v_ms = float(parsed_sentence.spd_over_grnd_kmph) / 3.6
        t = parsed_sentence.true_track

        vel_lng = float(v_ms) * math.cos(t)
        vel_lat = float(v_ms) * math.sin(t)

        vtg = {'t': t, 'v': v_ms, 'vlng': vel_lng, 'vlat': vel_lat}
        return vtg

    def _transform_gga(self, parsed_sentence):

        ts = parsed_sentence.timestamp
        time = (ts.hour * 60 * 60 * 1000) + (ts.minute * 60 * 1000) + (ts.second * 1000) + ts.microsecond

        lng_deg = int(float(parsed_sentence.lon) / 100)
        lng = lng_deg + (float(parsed_sentence.lon) - lng_deg * 100) / 60
        if (parsed_sentence.lon_dir == 'W'):
            lng = lng * -1

        lat_deg = int(float(parsed_sentence.lat) / 100)
        lat = lat_deg + (float(parsed_sentence.lat) - lat_deg * 100) / 60
        if (parsed_sentence.lat_dir == 'S'):
            lat = lat * -1

        self._get_ghashed_eov_coordinates(lng, lat)

        gga = {'time': time, 'lng': lng, 'lat': lat}
        return gga

    def get_gps_dictionary(self):
        readed_data = self.readed_data

        dict_of_methods = {
            "<class 'pynmea2.types.talker.GSA'>": self._transform_gsa,
            "<class 'pynmea2.types.talker.VTG'>": self._transform_vtg,
            "<class 'pynmea2.types.talker.GGA'>": self._transform_gga
        }

        for row in readed_data:
            if (('$GPGGA' in row) or ('$GPGSA' in row) or ('$GPVTG' in row)):
                parsed_sentence = pynmea2.parse(str(row))
                self.extracted_data.append(dict_of_methods[str(type(parsed_sentence))](parsed_sentence))

        def _transform_data_to_dictionary(extracted_data):

            gsa_list = self.extracted_data[0::3]
            vtg_list = self.extracted_data[1::3]
            gga_list = self.extracted_data[2::3]
            no_of_measurements = len(gsa_list)

            if (len(gsa_list) == len(vtg_list) == len(gga_list)):
                for i in range(no_of_measurements):
                    time_value = gga_list[i]['time']
                    self.measurement_dictionary[time_value] = {**gsa_list[i], **vtg_list[i], **gga_list[i]}
            else:
                print('Something\'s not right')

        _transform_data_to_dictionary(self.extracted_data)
        return self.measurement_dictionary

    def _get_ghashed_eov_coordinates(self, lng, lat):

        ghashed = Geohash.encode(lng, lat, precision=10)
        lng_to_tf, lat_to_tf = Geohash.decode(ghashed)
        in_proj = Proj(init='epsg:4326')
        out_proj = Proj(init='epsg:23700')

        if (not self.final_lng and not self.final_lat):
            lng, lat = transform(in_proj, out_proj, lng_to_tf, lat_to_tf)
            self.final_lng.append(lng)
            self.final_lat.append(lat)
            self.i = self.i + 1
        elif (lng_to_tf == self.final_lng[self.i] and lat_to_tf == self.final_lat[self.i]):
            pass
        else:
            lng, lat = transform(in_proj, out_proj, lng_to_tf, lat_to_tf)
            self.final_lng.append(lng)
            self.final_lat.append(lat)
            self.i = self.i + 1

        return {'lng': lng, 'lat': lat}
