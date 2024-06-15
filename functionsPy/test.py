import json
import unittest
import numpy as np
import tensorflow as tf

from scipy.stats import shapiro
from noise_interp import center_coordinate, truncated_normal, get_noise_interp_multi

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

    def test_linespace_interpolation_at_axis(self):
        # Define start and end tensors
        start_tensor = tf.constant([[0.0, 1.0], [2.0, 3.0]])  # Shape (2, 2)
        end_tensor = tf.constant([[10.0, 11.0], [12.0, 13.0]])  # Shape (2, 2)
        steps = 5
        axis = -2  # Generate linspace along the second to last axis

        # Calculate linspace with axis
        result = tf.linspace(start_tensor, end_tensor, steps, axis=axis)
        assert result.shape == (2, 5, 2)

    def test_get_noise_interp_multi(self):
        test = get_noise_interp_multi()
        print(test)
        print(test.shape)
