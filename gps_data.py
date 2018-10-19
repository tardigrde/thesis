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
        with open (filename, 'rt') as inputfile:
            self.readed_data = inputfile.readlines()
        
        self.extracted_data = []
        self.measurement_dictionary = {}

    def _transform_gsa(self, parsed_sentence):
        hdop = float(parsed_sentence.hdop)
        gsa = {'hdop': hdop}
        #self.sentence_info.append(gsa)
        return gsa

    def _transform_vtg(self, parsed_sentence):
        v = parsed_sentence.spd_over_grnd_kmph
        t = parsed_sentence.true_track
                        
        vel_lng = float(v) * math.cos(t)
        vel_lat = float(v) * math.sin(t)
                        
        vtg = {'t': t, 'v': v, 'vlng': vel_lng, 'vlat': vel_lat}
        #self.sentence_info.append(vtg)
        return vtg

    def _transform_gga(self, parsed_sentence):
        ts = parsed_sentence.timestamp
        time = (ts.hour * 60 * 60 * 1000 ) + (ts.minute * 60 * 1000 )+  (ts.second * 1000) + ts.microsecond
                                        
        lng_deg =int(float(parsed_sentence.lon)/100)
        lng = lng_deg + (float(parsed_sentence.lon) - lng_deg * 100)/60
        if(parsed_sentence.lon_dir == 'W'):
            lng = lng * -1
                        
        lat_deg = int(float(parsed_sentence.lat)/100)
        lat = lat_deg + (float(parsed_sentence.lat) - lat_deg * 100)/60
        if(parsed_sentence.lat_dir == 'S'):
            lat = lat * -1
                        
        gga = {'time': time, 'lng': lng, 'lat': lat}
        #self.sentence_info.append(gga)  
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

            if(len(gsa_list) == len(vtg_list) == len(gga_list)):
                for i in range(no_of_measurements):
                    time_value = gga_list[i]['time']
                    self.measurement_dictionary[time_value] = {**gsa_list[i], **vtg_list[i], **gga_list[i]}
            else:
                print('Something\'s not right')

        _transform_data_to_dictionary(self.extracted_data)
        return self.measurement_dictionary