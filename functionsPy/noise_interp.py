import tensorflow as tf

from constant import coord_depth, lat_len, coord_len

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


def center_coordinate(
        x
):  # allows to have sequences with even number length with anchor in the middle of the sequence
    return tf.reduce_mean(tf.stack([x, tf.roll(x, -1, -2)], 0), 0)[:, :-1, :]


def center_coordinate2(x):
    # Exclude the last element along the second dimension
    x_excluding_last = x[:, :-1, :]

    # Exclude the first element along the second dimension
    x_excluding_first = x[:, 1:, :]

    # Stack and reduce mean along the new dimension
    return tf.reduce_mean(tf.stack([x_excluding_last, x_excluding_first], 0), 0)

def truncated_normal(shape, bound=2.0):
    seed = tf.random.uniform((), minval=0, maxval=tf.int32.max, dtype=tf.int32)
    seeds = tf.random.experimental.stateless_split([seed, seed + 1], num=2)
    tensor = tf.random.stateless_truncated_normal(shape, seed=seeds[0], mean=0.0, stddev=1.0)
    clipped_tensor = tf.clip_by_value(tensor, clip_value_min=-bound, clip_value_max=bound)
    return clipped_tensor


def get_noise_interp():
    noiseg = tf.random.normal([1, 64], dtype=tf.float32)

    noisel = tf.concat([tf.random.normal([1, coord_depth], dtype=tf.float32), noiseg], -1)
    noisec = tf.concat([tf.random.normal([1, coord_depth], dtype=tf.float32), noiseg], -1)
    noiser = tf.concat([tf.random.normal([1, coord_depth], dtype=tf.float32), noiseg], -1)

    rl = tf.linspace(noisel, noisec, coord_len + 1, axis=-2)[:, :-1, :]
    rr = tf.linspace(noisec, noiser, coord_len + 1, axis=-2)

    noisetot = tf.concat([rl, rr], -2)
    noisetot = center_coordinate(noisetot)
    return crop_coordinate(noisetot)

def crop_coordinate(
        x
):  # randomly crops a conditioning sequence such that the anchor vector is at center of generator receptive field (maximum context is provided to the generator)
    fac = tf.random.uniform((), 0, coord_len // (lat_len // 2), dtype=tf.int32)

    def crop_case(offset):
        return tf.reshape(
            x[
            :,
            (lat_len // 4) + offset * (lat_len // 2): (lat_len // 4) + offset * (
                    lat_len // 2) + lat_len,
            :,
            ],
            [-1, lat_len, x.shape[-1]]
        )

    case_0 = lambda: crop_case(0)
    case_1 = lambda: crop_case(1)
    case_2 = lambda: crop_case(2)

    result = tf.cond(fac == 0, case_0, lambda: tf.cond(fac == 1, case_1, case_2))
    return result


def get_noise_interp_multi(fac=1, var=2.0):
    noiseg = truncated_normal([1, coord_depth], var)

    coordratio = coord_len // lat_len

    noisels_len = range(3 + ((fac - 1) // coordratio))
    noisels = [
        tf.concat([truncated_normal([1, 64], var), noiseg], -1)
        for i in noisels_len
    ]

    rls_len = range(len(noisels) - 1)
    rls = tf.concat(
        [
            tf.linspace(noisels[k], noisels[k + 1], coord_len + 1, axis=-2)[:, :-1, :]
            for k in rls_len
        ],
        -2,
    )

    rls = center_coordinate(rls)
    rls = rls[:, lat_len // 4:, :]
    rls = rls[:, : (rls.shape[-2] // lat_len) * lat_len, :]

    rls = tf.split(rls, rls.shape[-2] // lat_len, -2)

    return tf.concat(rls[:fac], 0)
