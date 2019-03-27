import numpy as np
from shapely.geometry import Point
from pathlib import Path
from point import Point


def get_dicts_of_measurements(acc, gps):
    points = get_points(acc, gps)


def get_points(acc, gps):
    """
    Here we are fusing acc and gps.
    Args:
        acc:
        gps:

    Returns:

    """
    points = []
    for time, lng, lat, vel, tt, hdop, a in zip(gps['time'], gps['la'], gps['ln'], gps['v'], gps['t'], gps['hdop'],
                                                acc):
        sublist = []
        for tm, ln, lt, v, t, hdp, acc_obj in zip(time, lng, lat, vel, tt, hdop, a):
            p = Point(tm, ln, lt, v, t, hdp, acc_obj)
            sublist.append(p)
        points.append(sublist)
    return points


def trim_and_sync_dataset(acc, gps):
    """
    We have to remove
    Args:
        acc:
        gps:

    Returns:

    """
    gps_time = gps['time']
    acc_time = acc['_time']
    if not (gps_time[0] > acc_time[0]) and (gps_time[len(gps_time) - 1] < acc_time[len(acc_time) - 1]):
        print('ERROR: gps started first or gps stopped last')
        return

    gps_msrmnt = trim_gps(gps)
    acc, gps = trim_and_sync_acc(acc, gps_msrmnt)
    # check_if_trim_sync_went_ok(acc, gps)
    return acc, gps


def check_gps(gps):
    assert len(list(gps.keys())) == 6


def trim_and_sync_acc(acc, gps):
    gps_time = gps['time']
    acc_time = acc['_time']
    acc['_time'] = round_acc_timestamps(acc_time, gps_time)

    gps = split_measurement_lists_at_void(gps)
    check_gps(gps)

    list_of_lists_of_intervals = get_valid_time_intervals(gps)
    check_intervals_length(gps, list_of_lists_of_intervals)

    formatted_acc_indices = get_formatted_acc_indices(list_of_lists_of_intervals, acc['_time'])
    check_intervals_length(gps, formatted_acc_indices)

    formatted_acc_lists_of_objects = get_formatted_lists_of_lists_of_acc_objcts(formatted_acc_indices, acc)
    check_intervals_length(gps, formatted_acc_lists_of_objects)

    return formatted_acc_lists_of_objects, gps


def get_formatted_lists_of_lists_of_acc_objcts(formatted_acc_indices, acc):
    formatted_acc_objects = []
    for lists_of_indices in formatted_acc_indices:
        sublist_of_acc_objects = []
        for indices in lists_of_indices:
            acc_objects = create_acc_object(indices, acc)
            sublist_of_acc_objects.append(acc_objects)
        formatted_acc_objects.append(sublist_of_acc_objects)

    return formatted_acc_objects


def create_acc_object(indices, acc):
    acc_object = {
        '_time': [],
        'north': [],
        'east': [],
        'down': [],
    }
    for i in indices:
        acc_object['_time'].append(acc['_time'][i])
        acc_object['north'].append(acc['north'][i])
        acc_object['east'].append(acc['east'][i])
        acc_object['down'].append(acc['down'][i])
    for key in list(acc.keys()):
        check_acc_indices(acc_object[key])
    return acc_object


def get_formatted_acc_indices(list_of_lists_of_intervals, acc_time):
    formatted_acc_indices = []
    for lists_of_intervals in list_of_lists_of_intervals:
        sublist_of_acc_indices = []
        for intervals in lists_of_intervals:
            acc_indexes = get_lists_in_interval(intervals, acc_time)
            sublist_of_acc_indices.append(acc_indexes)
        formatted_acc_indices.append(sublist_of_acc_indices)

    return formatted_acc_indices


def get_lists_in_interval(intervals, acc_time):
    indices = []
    for i, at in enumerate(acc_time):
        if intervals[0] < at < (intervals[1] + 1):
            indices.append(i)
    check_acc_indices(indices)
    return indices


def check_acc_indices(indices):
    min_max_length_of_acc_lists = set(range(90, 110))
    assert len(indices) in min_max_length_of_acc_lists


def get_valid_time_intervals(gps):
    intervals = []
    for time in gps['time']:
        intrvl = []
        for i, t in enumerate(time):
            tup = (t - 500, t + 500)
            intrvl.append(tup)
        intervals.append(intrvl)
    return intervals


def check_intervals_length(gps, intervals):
    assert len(gps['time']) == len(intervals)
    for t, i in zip(gps['time'], intervals):
        assert len(t) == len(i)


def round_acc_timestamps(acc_time, gps_time):
    for gt in gps_time:
        for i in range(len(acc_time)):
            try:
                at, at_next = acc_time[i], acc_time[i + 1]
                if gt == acc_time[i]:
                    pass
                elif at < gt < at_next:
                    diff_prior, diff_posterior = abs(gt - at), abs(at_next - gt)
                    if diff_prior <= diff_posterior:
                        acc_time[i] = gt
                    else:
                        acc_time[i + 1] = gt
                    break
            except IndexError:
                break
    # only with flatted gps_time it is possible to test
    check_rounded_acc_time(acc_time, gps_time)
    return acc_time


def check_rounded_acc_time(acc_time, gps_time):
    round_accs = []
    for a in acc_time:
        if a % 1000 == 0:
            round_accs.append(a)
    round_acc_set = set(sorted(round_accs))
    gps_set = set(sorted(gps_time))
    assert gps_set.issubset(round_accs)


def trim_gps(gps):
    columns = list(gps.keys())
    count = len(gps['time'])
    five_percent = int(round(count / 10, 0))

    for c in columns:
        if c in gps and len(gps[c]) == count:
            del (gps[c][:five_percent], gps[c][-five_percent:])
        else:
            print('ERROR: trimming gps fails!')
    return gps


def check_if_trim_sync_went_ok(acc, gps):
    for a, g in zip(acc['_time'], gps['time']):
        assert a[0] < g[0] and a[len(a) - 1] > g[len(g) - 1]
        print('OK')


def split_measurement_lists_at_void(gps):
    time = gps['time']
    last_msrmnt_ts = time[0]
    to_break = [0]
    no_measurment_intervals = []
    for i, t in enumerate(time):
        if t - last_msrmnt_ts > 4999:
            to_break.append(i)
            no_measurment_intervals.append([last_msrmnt_ts, t])
        last_msrmnt_ts = t
    for c in list(gps.keys()):
        l = gps[c]
        gps[c] = [l[i:j] for i, j in zip(to_break, to_break[1:] + [None])]
        gps[c] = remove_too_short_sublist(gps[c])
    return gps


def remove_too_short_sublist(list):
    indices = []
    for i, sublist in enumerate(list):
        if len(sublist) < 10:
            indices.append(i)
    for i in reversed(indices):
        del list[i]
    return list

# def remove_accleration_where_no_valid_gps(acc, gps):
#     gps_time = gps['time']
#     acc_time = acc['_time']
#
#     gps, no_measurement_intervals = split_measurement_lists_at_void(gps)
#     intervals = get_valid_time_intervals(gps)
#
#     #  TODO:
#     #  - if sublist of measurments is lower than 10 drop it
#
#     voidable = []
#     counter = 0
#     for void in no_measurement_intervals:
#         for t in range(len(acc_time)):
#             try:
#                 # # NEMJÓÓÓÓÓÓÓÓÓÓÓÓ
#                 # if acc_time[t] > void[0]:
#                 #     counter += 1
#                 #     if counter > 50:
#                 #         voidable.append(t)
#                 # el
#                 if acc_time[t] > void[0] and acc_time[t + 50] < void[1]:
#                     voidable.append(t)
#
#             except IndexError:
#                 continue
#         counter = 0
#
#     for i in reversed(voidable):
#         del acc['_time'][i]
#         del acc['east'][i]
#         del acc['north'][i]
#         del acc['down'][i]
#
#     acc, _ = split_measurement_lists_at_void(acc)
#     return acc, gps
#
#
# def interpolate_and_trim_data(acc_lists, gps_lists):
#     c = list(acc_lists.keys())
#
#     time, lng, lat, p, v = interpolate_gps_datalists(acc_lists['acc_time'], gps_lists)
#
#     dataset = {
#         'time': acc_lists['acc_time'],
#         'east': acc_lists['acc_east'],
#         'north': acc_lists['acc_north'],
#         'down': acc_lists['acc_down'],
#         'gps_time': time,
#         'lng': lng,
#         'lat': lat,
#         'hdop': p,
#         'vel': v,
#     }
#     columns = list(dataset.keys())
#     count = len(dataset['time'])
#     five_percent = int(round(count / 20, 0))
#
#     for c in columns:
#         # Checking if all the lists has the same size
#         if not len(dataset[c]) == count: return
#         count = len(dataset[c])
#         del (dataset[c][:five_percent], dataset[c][-five_percent:])
#
#     return dataset
#
#
# def interpolate_gps_datalists(time, gps_data):
#     gps_time = gps_data['time']
#     ln = gps_data['ln']
#     lt = gps_data['la']
#     hdop = gps_data['hdop']
#     v = gps_data['v']
#
#     gtime = np.interp(time, gps_time, gps_time).tolist()
#     lng = np.interp(time, gps_time, ln).tolist()
#     lat = np.interp(time, gps_time, lt).tolist()
#     p = np.interp(time, gps_time, hdop).tolist()
#     v = np.interp(time, gps_time, v).tolist()
#
#     return gtime, lng, lat, p, v
# def add_acc_to_points(points, acc):
#     sublist = 0
#     i = 0
#     j = 0
#     print(len(acc['_time'][0]))
#     print(len(acc['_time'][1]))
#     print(len(acc['_time'][2]))
#
#     indices = []
#     for sublists in points:
#         last = sublists[0].time
#         _indices = []
#         for p in sublists:
#             gps_t = p.time
#             if gps_t != last:
#                 _indices.append((gps_t + last) / 2)
#                 last = gps_t
#         indices.append(_indices)
#
#     for at, an, ae, ad, ind in zip(acc['_time'], acc['north'], acc['east'], acc['down'], indices):
#         acc = {
#             '_time': [],
#             'north': [],
#             'east': [],
#             'down': [],
#         }
#         # if not len(at)-1 == ind:
#         #     print('ERROR wtfFFFFFFFFFFFFFFFFFFFFFFFFFFF')
#         #     return
#         for t, n, e, d in zip(at, an, ae, ad):
#             acc['_time'].append(t)
#             acc['north'].append(n)
#             acc['east'].append(e)
#             acc['down'].append(d)
#             try:
#                 if t <= ind[i]:
#                     continue
#                 else:
#                     print('len of acc', len(acc['_time']))
#                     # this  looks OK so far
#                     # what you have to change is the amount of acc when there is
#                     # less than 5k gap but more than 1k
#                     # investigate this: trim_and_sync_acc !!!!!!!!
#                     if len(acc['_time']) > 200:
#                         print('dsads')
#                     points[sublist][i].acc = acc
#                     i += 1
#                     acc = {
#                         '_time': [],
#                         'north': [],
#                         'east': [],
#                         'down': [],
#                     }
#             except IndexError:
#                 print('IndexError')
#         print('len of lastacc', len(acc['_time']))
#         points[sublist][i].acc = acc
#         acc = {
#             '_time': [],
#             'north': [],
#             'east': [],
#             'down': [],
#         }
#         sublist += 1
#         i = 0
#     return points
######################################################################
#
#
#
#
# -> below functions for when I almost lost my data -.-
#
# import os
#
#
# def fix_shapes(dir):
#     for root, dirs, files in os.walk(dir):
#         dir = str(dir)
#         if files:
#             for file in files:
#                 size_of_file = os.stat(Path(root + '\\' + file)).st_size
#                 if '.dbf' in file and size_of_file > 0:
#                     if 'new' in file or 'osm' in file:
#                         break
#                     rewrite_shape_file(root, file)
#         elif not files and dirs:
#             for dirr in dirs:
#                 path = Path(root + '\\' + dirr)
#                 fix_shapes(path)
#
#
# def rewrite_shape_file(dir, file):
#     from dbfread import DBF
#     from pandas import DataFrame
#     import geopandas
#     path_of_file = Path(dir + '\\' + file)
#     dbf = DBF(path_of_file)
#     df = DataFrame(iter(dbf))
#     try:
#         df['Coordinates'] = list(zip(df.lng, df.lat))
#     except Exception as e:
#         print(e)
#         try:
#             df['Coordinates'] = list(zip(df.ln, df.lt))
#         except Exception as e:
#             print('MÁÁÁÁÁÁÁÁÁÁR MEGINT', e)
#
#     df['Coordinates'] = df['Coordinates'].apply(Point)
#     crs = {'init': 'epsg:23700'}
#     gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
#     pathh = Path(dir + '\\' + r'new' + file)
#     print(pathh)
#     gdf.to_file(driver='ESRI Shapefile', filename=pathh)
#
#
#
#
# ###################################################
# NOT NEEDED probably
# this is if you want to create points fom a line, and wan to measure points from those points...
# from dbfread import DBF
# from pandas import DataFrame
# import shapefile
#
#
# dbf = DBF(r'D:\thesis\data\szeged_osm_data\gis_osm_roads_free_1.shp',encoding='utf-8')
# frame = DataFrame(iter(dbf))
#
# print(frame)
# import os
#
# This is the half-way point along a great circle path between the two points.1
# Formula: 	Bx = cos φ2 ⋅ cos Δλ
# 	By = cos φ2 ⋅ sin Δλ
# 	φm = atan2( sin φ1 + sin φ2, √(cos φ1 + Bx)² + By² )
# 	λm = λ1 + atan2(By, cos(φ1)+Bx)
# JavaScript:
# (all angles
# in radians)
#
#
# var Bx = Math.cos(φ2) * Math.cos(λ2-λ1);
# var By = Math.cos(φ2) * Math.sin(λ2-λ1);
# var φ3 = Math.atan2(Math.sin(φ1) + Math.sin(φ2),
#                     Math.sqrt( (Math.cos(φ1)+Bx)*(Math.cos(φ1)+Bx) + By*By ) );
# var λ3 = λ1 + Math.atan2(By, Math.cos(φ1) + Bx);
#
# 	The longitude can be normalised to −180…+180 using (lon+540)%360-180
#
# Just as the initial bearing may vary from the final bearing, the midpoint may not be located half-way between latitudes/longitudes; the midpoint between 35°N,45°E and 35°N,135°E is around 45°N,90°E.
# Intermediate point
#
# An intermediate point at any fraction along the great circle path between two points can also be calculated.1
# Formula: 	a = sin((1−f)⋅δ) / sin δ
# 	b = sin(f⋅δ) / sin δ
# 	x = a ⋅ cos φ1 ⋅ cos λ1 + b ⋅ cos φ2 ⋅ cos λ2
# 	y = a ⋅ cos φ1 ⋅ sin λ1 + b ⋅ cos φ2 ⋅ sin λ2
# 	z = a ⋅ sin φ1 + b ⋅ sin φ2
# 	φi = atan2(z, √x² + y²)
# 	λi = atan2(y, x)
# where 	f is fraction along great circle route (f=0 is point 1, f=1 is point 2), δ is the angular distance d/R between the two points.


#  * Returns the point at given fraction between ‘this’ point and given point.
#  *
#  * @param   {LatLon} point - Latitude/longitude of destination point.
#  * @param   {number} fraction - Fraction between the two points (0 = this point, 1 = specified point).
#  * @returns {LatLon} Intermediate point between this point and destination point.
#  *
#  * @example
#  *   const p1 = new LatLon(52.205, 0.119);
#  *   const p2 = new LatLon(48.857, 2.351);
#  *   const pInt = p1.intermediatePointTo(p2, 0.25); // 51.3721°N, 000.7073°E
#
# intermediatePointTo(point, fraction) {
#     if (!(point instanceof LatLonSpherical)) point = LatLonSpherical.parse(point); // allow literal forms
#
#     const φ1 = this.lat.toRadians(), λ1 = this.lon.toRadians();
#     const φ2 = point.lat.toRadians(), λ2 = point.lon.toRadians();
#
#     // distance between points
#     const Δφ = φ2 - φ1;
#     const Δλ = λ2 - λ1;
#     const a = Math.sin(Δφ/2) * Math.sin(Δφ/2)
#         + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ/2) * Math.sin(Δλ/2);
#     const δ = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
#
#     const A = Math.sin((1-fraction)*δ) / Math.sin(δ);
#     const B = Math.sin(fraction*δ) / Math.sin(δ);
#
#     const x = A * Math.cos(φ1) * Math.cos(λ1) + B * Math.cos(φ2) * Math.cos(λ2);
#     const y = A * Math.cos(φ1) * Math.sin(λ1) + B * Math.cos(φ2) * Math.sin(λ2);
#     const z = A * Math.sin(φ1) + B * Math.sin(φ2);
#
#     const φ3 = Math.atan2(z, Math.sqrt(x*x + y*y));
#     const λ3 = Math.atan2(y, x);
#
#     const lat = φ3.toDegrees();
#     const lon = λ3.toDegrees();
#
#     return new LatLonSpherical(lat, Dms.wrap180(lon));
# }
