class Point:
    def __init__(self, time, lng, lat, vel, bear, hdop, acc):
        self._lng = lng
        self._lat = lat
        self._time = time
        self._vel = vel
        self._bear = bear
        self._hdop = hdop
        self._acc = acc

    def __repr__(self):
        return "{}: ({}, {})".format(self.time,self.lng,self.lat)

    @property
    def time(self):
        return self._time

    @property
    def lng(self):
        return self._lng

    @property
    def lat(self):
        return self._lat

    @property
    def vel(self):
        return self._vel

    @property
    def bear(self):
        return self._bear

    @property
    def hdop(self):
        return self._hdop

    @property
    def acc(self):
        return self._acc

