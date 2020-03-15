class Sensor:
    def __init__(self):
        pass

    def read_data(self):
        raise NotImplementedError()


class InertialMeasurementUnit(Sensor):
    def __init__(self):
        super().__init__()
        self.acc = None
        self.gyro = None
        self.mag = None


class GlobalPositioningSystem(Sensor):
    def __init__(self):
        super().__init__()


class Measurement:
    def __init__(self):
        pass


class IMUMeasurement(Measurement):
    def __init__(self):
        super().__init__()


class GPSMeasurement(Measurement):
    def __init__(self):
        super().__init__()
