import tensorflow as tf
import librosa

from constant import sigma_rescale, mu_rescale, hop, sr, mel_bins

melmat = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=mel_bins,
    num_spectrogram_bins=(4 * hop * 2) // 2 + 1,
    sample_rate=sr,
    lower_edge_hertz=0.0,
    upper_edge_hertz=sr // 2,
)
mel_f = tf.convert_to_tensor(librosa.mel_frequencies(n_mels=mel_bins + 2, fmin=0.0, fmax=sr // 2))
enorm = tf.cast(
    tf.expand_dims(
        tf.constant(2.0 / (mel_f[2: mel_bins + 2] - mel_f[: mel_bins])),
        0,
    ),
    tf.float32,
)
melmat = tf.multiply(melmat, enorm)
melmat = tf.divide(melmat, tf.reduce_sum(melmat, axis=0))
melmat = tf.where(tf.math.is_nan(melmat), tf.zeros_like(melmat), melmat)


def wv2spec_hop(wv, topdb=80.0, hopsize=256):
    X = tf.signal.stft(
        wv,
        frame_length=4 * hopsize,
        frame_step=hopsize,
        fft_length=4 * hopsize,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    S = normalize(power2db(tf.abs(X) ** 2, top_db=topdb))
    return tf.tensordot(S, melmat, 1)


def normalize(S, clip=False):
    S = (S - mu_rescale) / sigma_rescale
    if clip:
        S = tf.clip_by_value(S, -1.0, 1.0)
    return S


def power2db(power, ref_value=1.0, amin=1e-10, top_db=None, norm=False):
    log_spec = 10.0 * _tf_log10(tf.maximum(amin, power))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
    if top_db is not None:
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
    return log_spec


def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
