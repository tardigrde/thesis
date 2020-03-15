from yaml import dump, Dumper, load, FullLoader


class Pipeline:
    def __init__(self):
        self.workspace = self.get_workspace()
        self.paths = self.read_paths_of_data_files()

    def read_paths_of_data_files(self):
        return read_yaml('paths.yml')

    def get_workspace(self):
        import os
        return os.path.dirname(os.path.realpath(__file__))

    def get_imu_data(self):  # still unprocessed
        pass

    def get_gps_data(self):  # still unprocessed
        pass

    def pre_process(self):
        pass

    def fuse_data(self):
        pass

    def apply_kalman_filter(self):
        # here it is important to be able to easily switch between this or that or adaptive kalman filtering
        pass

    def evaluate(self):
        pass

    def create_outputs(self):
        pass


def read_yaml(file):
    try:
        with open(file) as yaml:
            return load(yaml, Loader=FullLoader)
    except FileNotFoundError:
        print('File does not exists!')


def dump_to_file(file, data):
    with open(file, 'w') as f:
        dump(data, f, Dumper=Dumper)
    print(f"Saved data to {file}!")
