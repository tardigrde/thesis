class ImuMeasurement:
    def __init__(self, data):
        self.data = data
        self.fill_values(data)

    def fill_values(self, data):
        pass

    @property
    def acc_z(self):
        return _acc