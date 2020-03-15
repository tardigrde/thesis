import unittest


class TestImuPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.gm = '/home/levente/projects/thesis/tests/kalmaned/adapted_gm.csv'
        self.aprtc = '/home/levente/projects/thesis/tests/kalmaned/adapted.csv'

    def test_same_csv_file(self):
        with open(self.gm, mode='r') as gm_file, open(self.aprtc, mode='r') as aprtc_file:
            for gm_line, aprtc_line in zip(gm_file, aprtc_file):
                self.assertEqual(gm_line, aprtc_line)
