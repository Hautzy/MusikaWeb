import json
import unittest
import numpy as np
import tensorflow as tf

from scipy.stats import shapiro
from noise_interp import center_coordinate, truncated_normal

TEST_FOLDER = 'test_input'


class TestFunctions(unittest.TestCase):
    def test_center_coordinate(self):
        test_tensor = tf.constant([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
        result = center_coordinate(test_tensor)

        with open(f'{TEST_FOLDER}/testCenterCoordinate.json', 'r') as file:
            result_array = json.load(file)
            result_tensor = tf.constant(result_array)

        assert np.all(tf.equal(result, result_tensor).numpy())

    def run_normal_test(self, result):
        # Perform Shapiro-Wilk test
        stat, p = shapiro(result.numpy())
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Sample looks Gaussian (fail to reject H0)')
            assert True
        else:
            print('Sample does not look Gaussian (reject H0)')
            assert False

    def test_truncated_normal(self):
        shape = [9, 9]
        bound = 2.0
        result = truncated_normal(shape, bound)
        print(result)
        self.run_normal_test(result)

        with open(f'{TEST_FOLDER}/testTruncatedNormal.json', 'r') as file:
            result_array = json.load(file)
            result_tensor = tf.constant(result_array)
            self.run_normal_test(result_tensor)


