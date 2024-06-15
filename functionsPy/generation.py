import numpy as np
import tensorflow as tf

from constant import lat_depth, mu_rescale, sigma_rescale, hop, shape


def distribute_gen(x, model, bs=32):
    outls = []
    bdim = x.shape[0]
    if bdim == 1:
        bdim = 2
    for i in range(((bdim - 2) // bs) + 1):
        outls.append(model(x[i * bs: i * bs + bs], training=False))
    return tf.concat(outls, 0)


def distribute_dec2(x, model, bs=32):
    outls = []
    bdim = x.shape[0]
    for i in range(((bdim - 2) // bs) + 1):
        inp1 = x[i * bs: i * bs + bs]
        inpls = tf.split(inp1, 2, -2)
        inp1 = tf.concat(inpls, 0)
        outls.append(model(inp1, training=False))

    return tf.concat(outls, 0)


def distribute_dec(x, model, bs=32):
    outls = []
    bdim = x.shape[0]
    for i in range(((bdim - 2) // bs) + 1):
        inp = x[i * bs: i * bs + bs]
        inpls = tf.split(inp, 2, -2)
        inp = tf.concat(inpls, 0)
        res = model(inp, training=False)
        outls.append(res)
    return np.concatenate([outls[k][0] for k in range(len(outls))], 0), np.concatenate(
        [outls[k][1] for k in range(len(outls))], 0
    )


def denormalize(S, clip=False):
    if clip:
        S = tf.clip_by_value(S, -1.0, 1.0)
    return (S * sigma_rescale) + mu_rescale


def db2power(S_db, ref=1.0):
    return ref * tf.math.pow(10.0, 0.1 * S_db)


def conc_tog_specphase(S, P):
    S = tf.cast(S, tf.float32)
    P = tf.cast(P, tf.float32)
    S = denormalize(S, clip=False)
    S = tf.math.sqrt(db2power(S) + 1e-7)
    P = P * np.pi
    Sls = tf.split(S, S.shape[0], 0)
    S = tf.squeeze(tf.concat(Sls, 1), 0)
    Pls = tf.split(P, P.shape[0], 0)
    P = tf.squeeze(tf.concat(Pls, 1), 0)
    SP = tf.cast(S, tf.complex64) * tf.math.exp(1j * tf.cast(P, tf.complex64))
    wv = tf.signal.inverse_stft(
        SP,
        4 * hop,
        hop,
        fft_length=4 * hop,
        window_fn=tf.signal.inverse_stft_window_fn(hop),
    )
    return np.squeeze(wv)


def generate_waveform(inp, gen_ema, dec, dec2, batch_size=64):
    ab = distribute_gen(inp, gen_ema, bs=batch_size)
    abls = tf.split(ab, ab.shape[0], 0)
    ab = tf.concat(abls, -2)
    abls = tf.split(ab, ab.shape[-2] // 8, -2)
    abi = tf.concat(abls, 0)

    chls = []
    for channel in range(2):
        ab = distribute_dec2(
            abi[:, :, :, channel * lat_depth: channel * lat_depth + lat_depth],
            dec2,
            bs=batch_size,
        )
        abls = tf.split(ab, ab.shape[-2] // shape, -2)
        ab = tf.concat(abls, 0)

        ab_m, ab_p = distribute_dec(ab, dec, bs=batch_size)
        abwv = conc_tog_specphase(ab_m, ab_p)
        chls.append(abwv)

    return np.clip(np.squeeze(np.stack(chls, -1)), -1.0, 1.0)


def generate_example_stereo(self, gen_ema, dec, dec2):
    abb = gen_ema(self.get_noise_interp(), training=False)
    abbls = tf.split(abb, abb.shape[-2] // 8, -2)
    abb = tf.concat(abbls, 0)

    chls = []
    for channel in range(2):
        ab = self.distribute_dec2(
            abb[
            :,
            :,
            :,
            channel * self.args.latdepth: channel * self.args.latdepth + self.args.latdepth,
            ],
            dec2,
        )
        abls = tf.split(ab, ab.shape[-2] // self.args.shape, -2)
        ab = tf.concat(abls, 0)
        ab_m, ab_p = self.distribute_dec(ab, dec)
        wv = self.conc_tog_specphase(ab_m, ab_p)
        chls.append(wv)

    return np.stack(chls, -1)
