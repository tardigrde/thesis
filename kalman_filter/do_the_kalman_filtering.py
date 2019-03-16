from kalman_filter.archive.interpolation.constant_acceleration import interface


def do_kalmaning_with_interpolation(acc, gps, fig_dir):
    result = interface.get_kalmaned_datatable(acc, gps, fig_dir)
    return result