import numpy as np
from shapely.geometry import Point
from pathlib import Path
from point import Point


def get_dicts_of_measurements(acc, gps):
    points = get_points(gps)
    points_with_acc = add_acc_to_points(points, acc)


def get_points(gps):
    points = []
    for time, lng, lat, vel, tt, hdop in zip(gps['time'], gps['la'], gps['ln'], gps['v'], gps['t'], gps['hdop']):
        sublist = []
        for tm, ln, lt, v, t, hdp in zip(time, lng, lat, vel, tt, hdop):
            p = Point(tm, ln, lt, v, t, hdp)
            sublist.append(p)
        points.append(sublist)
    return points


def add_acc_to_points(points, acc):
    sublist = 0
    i = 0
    j = 0
    print(len(acc['_time'][0]))
    print(len(acc['_time'][1]))
    print(len(acc['_time'][2]))

    indices=[]
    for sublists in points:
        last = sublists[0].time
        _indices=[]
        for  p in sublists:
            gps_t = p.time
            if gps_t != last:
                _indices.append((gps_t+last)/2)
                last = gps_t
        indices.append(_indices)


    for at, an, ae, ad, ind in zip(acc['_time'], acc['north'], acc['east'], acc['down'], indices):
        acc = {
            '_time': [],
            'north': [],
            'east': [],
            'down': [],
        }
        # if not len(at)-1 == ind:
        #     print('ERROR wtfFFFFFFFFFFFFFFFFFFFFFFFFFFF')
        #     return
        for t, n, e, d in zip(at, an, ae, ad):
            acc['_time'].append(t)
            acc['north'].append(n)
            acc['east'].append(e)
            acc['down'].append(d)
            try:
                if t<=ind[i]:
                    continue
                else:
                    print('len of acc',len(acc['_time']))
                    # this  looks OK so far
                    # what you have to change is the amount of acc when there is
                    # less than 5k gap but more than 1k
                    # investigate this: trim_and_sync_acc !!!!!!!!
                    if len(acc['_time']) > 200:
                        print('dsads')
                    points[sublist][i].acc = acc
                    i += 1
                    acc = {
                        '_time': [],
                        'north': [],
                        'east': [],
                        'down': [],
                    }
            except IndexError:
                print('IndexError')
        print('len of lastacc', len(acc['_time']))
        points[sublist][i].acc = acc
        acc = {
            '_time': [],
            'north': [],
            'east': [],
            'down': [],
        }
        sublist+=1
        i = 0

    return points


def trim_and_sync_dataset(acc_msrmnt, gps_msrmnt):
    gps_msrmnt = trim_gps(gps_msrmnt)
    acc, gps = trim_and_sync_acc(acc_msrmnt, gps_msrmnt)
    return acc, gps


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


def trim_and_sync_acc(acc, gps):
    gps_time = gps['time']
    # TODO:
    # - maybe trim very low speed gps-es
    first_measurement, last_measurement = gps_time[0], gps_time[len(gps_time) - 1]
    at = acc['_time']
    a_first_index, a_last_index = 0, 0

    for i in range(len(at)):
        try:
            if first_measurement >= at[i] and first_measurement <= at[i + 1]:
                a_first_index = i
            elif last_measurement >= at[i] and last_measurement <= at[i + 1]:
                a_last_index = i + 1
        except IndexError as error:
            a_last_index = i
            break
    to_trim = [(a_first_index - 1), (len(at) - a_last_index - 1)]

    columns = list(acc.keys())
    for c in columns:
        del (acc[c][:to_trim[0]], acc[c][-to_trim[1]:])

    acc['_time'] = round_acc_timestamps(acc['_time'], gps_time)
    acc, gps = remove_accleration_where_no_valid_gps(acc, gps)
    return acc, gps


def split_measurement_lists_at_void(data):
    time = data['time'] if 'time' in list(data.keys()) else data['_time']
    last_msrmnt_ts = time[0]
    to_break = [0]
    no_measurment_intervals = []
    for i, t in enumerate(time):
        if t - last_msrmnt_ts > 4999:
            to_break.append(i)
            no_measurment_intervals.append([last_msrmnt_ts, t])
        last_msrmnt_ts = t
    for c in list(data.keys()):
        l = data[c]
        data[c] = [l[i:j] for i, j in zip(to_break, to_break[1:] + [None])]
    return data, no_measurment_intervals


def remove_accleration_where_no_valid_gps(acc, gps):
    gps_time = gps['time']
    acc_time = acc['_time']

    gps, no_measurement_intervals = split_measurement_lists_at_void(gps)

    voidable = []
    for void in no_measurement_intervals:
        for t in range(len(acc_time)):
            if acc_time[t] > void[0] and acc_time[t] < void[1]:
                voidable.append(t)

    for i in reversed(voidable):
        del acc['_time'][i]
        del acc['east'][i]
        del acc['north'][i]
        del acc['down'][i]

    acc, _ = split_measurement_lists_at_void(acc)
    return acc, gps


def round_acc_timestamps(acc_time, gps_time):
    for gt in gps_time:
        last_gps_ts = gt
        for i in range(len(acc_time)):
            try:
                at = acc_time[i]
                at_post = acc_time[i + 1]
                if gt == acc_time[i]:
                    # print('Lucky you. They are the same: ', gt, ' = ', acc_time[i])
                    break
                if gt > acc_time[i] and gt < acc_time[i + 1]:
                    diff_prior = abs(gt - acc_time[i])
                    diff_posterior = abs(gt - acc_time[i + 1])
                    if diff_prior <= diff_posterior:
                        # both if prior is smaller and if prior and posterior are the same, prior should equal the gt's ts
                        acc_time[i] = gt
                    else:
                        acc_time[i + 1] = gt
                    break
            except IndexError as error:
                # print('index out of range: ', error)
                break
    print(gps_time)
    i = []
    for a in acc_time:
        if a % 1000 == 0:
            i.append(a)
    print('acc ', i)
    print(gps_time[0], gps_time[len(gps_time) - 1])
    print(acc_time[0], acc_time[len(acc_time) - 1])
    return acc_time


def interpolate_and_trim_data(acc_lists, gps_lists):
    c = list(acc_lists.keys())

    time, lng, lat, p, v = interpolate_gps_datalists(acc_lists['acc_time'], gps_lists)

    dataset = {
        'time': acc_lists['acc_time'],
        'east': acc_lists['acc_east'],
        'north': acc_lists['acc_north'],
        'down': acc_lists['acc_down'],
        'gps_time': time,
        'lng': lng,
        'lat': lat,
        'hdop': p,
        'vel': v,
    }
    columns = list(dataset.keys())
    count = len(dataset['time'])
    five_percent = int(round(count / 20, 0))

    for c in columns:
        # Checking if all the lists has the same size
        if not len(dataset[c]) == count: return
        count = len(dataset[c])
        del (dataset[c][:five_percent], dataset[c][-five_percent:])

    return dataset


def interpolate_gps_datalists(time, gps_data):
    gps_time = gps_data['time']
    ln = gps_data['ln']
    lt = gps_data['la']
    hdop = gps_data['hdop']
    v = gps_data['v']

    gtime = np.interp(time, gps_time, gps_time).tolist()
    lng = np.interp(time, gps_time, ln).tolist()
    lat = np.interp(time, gps_time, lt).tolist()
    p = np.interp(time, gps_time, hdop).tolist()
    v = np.interp(time, gps_time, v).tolist()

    return gtime, lng, lat, p, v


def _pass_std_devs(acc):
    std_dev_acc_east = np.std(acc['acc_east'])
    std_dev_acc_north = np.std(acc['acc_north'])
    return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}


"""
below functions for when I almost lost my data -.-
"""
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

# NOT NEEDED probably
# this is if you want to create points fom a line, and wan to measure points from those points...
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
