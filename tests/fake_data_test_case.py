import numpy as np
import unittest
from tempfile import TemporaryFile

from DL.utils.data_loading import loadRobotData


# TODO: close the fake data file at the end.
class FakeDataTestCase(unittest.TestCase):

    def setUp(self):
        self.fake_data_npzfile = TemporaryFile()
        data_keys = ['measured_angles', 'measured_velocities', 'measured_torques',
                'constrained_torques']
        fake_data_dict = {}
        nseq = 5
        seq_length = 1000
        for k in data_keys:
            fake_data_dict[k] = np.random.rand(nseq, seq_length, 3)
        np.savez(self.fake_data_npzfile, **fake_data_dict)

        # File resets needed for subsequent reading.
        _ = self.fake_data_npzfile.seek(0)
        self.observations, self.actions = loadRobotData(self.fake_data_npzfile)
        _ = self.fake_data_npzfile.seek(0)
