import json
import unittest
import numpy as np
import tensorflow as tf

from noise_interp import center_coordinate

TEST_FOLDER = 'test_input'


class TestFunctions(unittest.TestCase):
    def test_center_coordinate(self):
        test_tensor = tf.constant([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
        result = center_coordinate(test_tensor)

        with open(f'{TEST_FOLDER}/output.json', 'r') as file:
            result_array = json.load(file)
            result_tensor = tf.constant(result_array)

        assert np.all(tf.equal(result, result_tensor).numpy())
