import tensorflow as tf
from tensorflow.python.framework import random_seed

'''
args

coordepth 64 Dimension of latent coordinate and style random vectors
coordlen & latlen
    if args.small:
        args.latlen = 128
    else:
        args.latlen = 256
    args.coordlen = (args.latlen // 2) * 3
'''

coord_depth = 64
lat_len = 128
coord_len = (lat_len // 2) * 3


def center_coordinate(
        x
):  # allows to have sequences with even number length with anchor in the middle of the sequence
    return tf.reduce_mean(tf.stack([x, tf.roll(x, -1, -2)], 0), 0)[:, :-1, :]


def truncated_normal(shape, bound=2.0, dtype=tf.float32):
    seed1, seed2 = random_seed.get_seed(tf.random.uniform((), tf.int32.min, tf.int32.max, dtype=tf.int32))
    return tf.random.stateless_parameterized_truncated_normal(shape, [seed1, seed2], 0.0, 1.0, -bound, bound)


def get_noise_interp_multi(fac=1, var=2.0):
    noiseg = truncated_normal([1, coord_depth], var, dtype=tf.float32)

    coordratio = coord_len // lat_len

    noisels = [
        tf.concat([truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1)
        for i in range(3 + ((fac - 1) // coordratio))
    ]
    rls = tf.concat(
        [
            tf.linspace(noisels[k], noisels[k + 1], coord_len + 1, axis=-2)[:, :-1, :]
            for k in range(len(noisels) - 1)
        ],
        -2,
    )

    rls = center_coordinate(rls)
    rls = rls[:, lat_len // 4:, :]
    rls = rls[:, : (rls.shape[-2] // lat_len) * lat_len, :]

    rls = tf.split(rls, rls.shape[-2] // lat_len, -2)

    return tf.concat(rls[:fac], 0)



