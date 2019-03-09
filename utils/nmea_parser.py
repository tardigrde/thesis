from pyproj import Proj, transform
import pandas as pd
import Geohash
import pynmea2
import math


def _read_log_file(path):
    """

    Args:
        path:

    Returns:

    """
    with open(path, 'rt') as inputfile:
        read_data = inputfile.readlines()
    return read_data


def _transform_gsa(parsed_sentence):
    """

    Args:
        parsed_sentence:

    Returns:

    """

    hdop = float(parsed_sentence.hdop)
    gsa = {'hdop': hdop}
    return gsa


def _transform_vtg(parsed_sentence):
    """

    Args:
        parsed_sentence:

    Returns:

    """
    v_ms = float(parsed_sentence.spd_over_grnd_kmph) / 3.6
    t = parsed_sentence.true_track

    vel_lng = float(v_ms) * math.cos(t)
    vel_lat = float(v_ms) * math.sin(t)

    vtg = {'t': t, 'v': int(v_ms), 'vlng': vel_lng, 'vlat': vel_lat}
    return vtg


def _transform_gga(parsed_sentence):
    """

    Args:
        parsed_sentence:

    Returns:

    """
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

    gga = {'time': time, 'lng': lng, 'lat': lat}
    return gga


def _transform_data_to_dictionary(extracted_data):
    """

    Args:
        extracted_data:

    Returns:

    """
    measurement_dictionary = {}

    gsa_list = extracted_data[0::3]
    vtg_list = extracted_data[1::3]
    gga_list = extracted_data[2::3]

    no_of_measurements = len(gga_list)

    if len(gsa_list) == len(vtg_list) == len(gga_list):
        for i in range(no_of_measurements):
            time_value = gga_list[i]['time']
            measurement_dictionary[time_value] = {**gsa_list[i], **vtg_list[i], **gga_list[i]}
    else:
        print("Something's not right")

    return measurement_dictionary


def _get_ghashed_eov_coordinates(lng, lat):
    """

    Args:
        lng:
        lat:

    Returns:

    """
    ghashed = Geohash.encode(lng, lat, precision=11)
    lng_to_tf, lat_to_tf = Geohash.decode(ghashed)
    return {'lng': lng_to_tf, 'lat': lat_to_tf}


def _remove_redundant_points(msrmnt_dict):
    """

    Args:
        msrmnt_dict:

    Returns:

    """
    in_proj = Proj(init='epsg:4326')
    out_proj = Proj(init='epsg:23700')
    time_list = sorted(list(msrmnt_dict.keys()))
    no_of_measurements = len(time_list)
    print('Original gps measurment count was: {}'.format(len(time_list)))
    list_of_dicts_of_gps_data = []
    gps_data_dict = {}
    counter = -1

    for i in range(no_of_measurements):
        m = msrmnt_dict[time_list[i]]
        ln, lt = m['lng'], m['lat']
        coords = _get_ghashed_eov_coordinates(ln, lt)
        lng, lat = transform(in_proj, out_proj, coords['lng'], coords['lat'])

        measurment_attributes = {
            'time': m['time'], 'hdop': m['hdop'],
            'vlng': m['vlng'], 'vlat': m['vlat'],
            't': m['t'], 'v': m['v'],
            'lng': lng, 'lat': lat,
        }
        # print('THIS LOK AT THIS', measurment_attributes)
        """
        TODO:
        - refactor list_of_dicts_of_gps_data
        """
        if not list_of_dicts_of_gps_data:
            list_of_dicts_of_gps_data.append(measurment_attributes)
            gps_data_dict[m['time']] = measurment_attributes
            counter = counter + 1
        elif lng == list_of_dicts_of_gps_data[counter]['lng'] and lat == list_of_dicts_of_gps_data[counter]['lat']:
            pass
        else:
            list_of_dicts_of_gps_data.append(measurment_attributes)
            gps_data_dict[m['time']] = measurment_attributes
            counter = counter + 1
    return gps_data_dict


def get_gps_dataframe(gps_data_dict):
    """

    Args:
        gps_data_dict:

    Returns:

    """
    gps_dataframe = pd.DataFrame(gps_data_dict)
    return gps_dataframe


def pass_gps_dict_of_lists(gps):
    time, ln, la, v, t, hdop = [], [], [], [], [], []
    timestamps = sorted(list(gps.keys()))
    for t in timestamps:
        values = gps[t]
        v.append(values['v'])
        hdop.append(values['hdop'])
        time.append(values['time'])
        ln.append(values['lng'])
        la.append(values['lat'])
    return {'ln': ln, 'la': la, 'hdop': hdop, 'time': time, 'v':v }


# Call this and on the result of this you can call get_gps_dataframe.
def get_gps_dictionary(path, data='lists'):
    """

    Args:
        path:

    Returns:

    """
    read_data = _read_log_file(path)

    extracted_data = []

    dict_of_methods = {
        "<class 'pynmea2.types.talker.GSA'>": _transform_gsa,
        "<class 'pynmea2.types.talker.VTG'>": _transform_vtg,
        "<class 'pynmea2.types.talker.GGA'>": _transform_gga,
    }

    for row in read_data:
        if (('$GPGGA' in row) or ('$GPGSA' in row) or ('$GPVTG' in row)):
            parsed_sentence = pynmea2.parse(str(row))
            extracted_data.append(dict_of_methods[str(type(parsed_sentence))](parsed_sentence))

    msrmnt_dict = _transform_data_to_dictionary(extracted_data)
    gps_data_dict = _remove_redundant_points(msrmnt_dict)
    
    if data == 'lists':
        return pass_gps_dict_of_lists(gps_data_dict)
    elif data == 'df':
        return get_gps_dataframe(gps_data_dict)
    elif data == 'dicts':
        return gps_data_dict
