import json
import unittest
import numpy as np
import tensorflow as tf

from generation import denormalize, db2power, conc_tog_specphase

TEST_FOLDER = 'test_input'


class TestGeneration(unittest.TestCase):
    def test_denormalize_unclip(self):
        S = tf.constant([0.5, -0.5, 0.0])
        result = denormalize(S)
        print(result)

    def test_denormalize_clip(self):
        S = tf.constant([0.5, -0.5, 0.0])
        result = denormalize(S, clip=True)
        print(result)

    def test_db2power(self):
        S = tf.constant([2.0, 4.0, 0.0])
        result = db2power(S)
        print(result)

    def test_conc_tog_specphase(self):
        S = tf.constant([[[0.5, -0.5, 0.0], [0.3, -0.3, 0.2]]], dtype=tf.float32)
        P = tf.constant([[[0.1, -0.1, 0.2], [0.4, -0.4, 0.5]]], dtype=tf.float32)

        result = conc_tog_specphase(S, P)
        print(result)