class Point:
    def __init__(self, time, lng, lat, vel, bear, hdop, acc):
        self.lng = lng
        self.lat = lat
        self.time = time
        self.vel = vel
        self.bear = bear
        self.hdop = hdop
        self.acc = acc

    def __repr__(self):
        pass

    def get_attribute(self, type):
        if type == 'time':
            return self.time
