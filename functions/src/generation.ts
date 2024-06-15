import * as tf from '@tensorflow/tfjs';
import {MU_RESCALE, SIGMA_RESCALE} from "./constant";

export function denormalize(S: tf.Tensor, clip=false): tf.Tensor {
    if (clip) {
        S = S.clipByValue(-1.0, 1.0);
    }
    return S.mul(SIGMA_RESCALE).add(MU_RESCALE);
}

export function db2power(S_db: tf.Tensor, ref=1.0) {
    const refTensor = tf.scalar(ref);
    const power = tf.scalar(10.0).pow(tf.scalar(0.1).mul(S_db));
    return refTensor.mul(power);
}



/*
function createHannWindow(length: number) {
    const n = tf.range(0, length);
    const window = 0.5 - 0.5 * tf.cos(2 * Math.PI * n.div(length - 1));
    return window;
}

async function inverse_stft(SP, frameLength, frameStep, fftLength, windowLength) {
    const hannWindow = createHannWindow(windowLength);

    // Inverse FFT on each frame
    const frames = tf.tidy(() => {
        const inverseFft = tf.spectral.ifft(SP);
        const realFrames = tf.real(inverseFft);
        const windowedFrames = realFrames.mul(hannWindow);
        return windowedFrames;
    });

    // Overlap-add
    const numFrames = frames.shape[0];
    const signalLength = numFrames * frameStep + (frameLength - frameStep);
    const signal = tf.buffer([signalLength], 'float32');

    for (let i = 0; i < numFrames; i++) {
        const start = i * frameStep;
        const frame = frames.slice([i, 0], [1, frameLength]).reshape([frameLength]);
        signal.set(frame, start);
    }

    return signal.toTensor();
}

async function conc_tog_specphase(S: tf.Tensor, P: tf.Tensor) {
    S = tf.cast(S, 'float32');
    P = tf.cast(P, 'float32');
    S = denormalize(S, false);
    S = tf.sqrt(db2power(S).add(tf.scalar(1e-7)));
    P = P.mul(Math.PI);

    const Sls = tf.split(S, S.shape[0], 0);
    S = tf.squeeze(tf.concat(Sls, 1), [0]);
    const Pls = tf.split(P, P.shape[0], 0);
    P = tf.squeeze(tf.concat(Pls, 1), [0]);

    const SP = tf.cast(S, 'complex64').mul(tf.exp(tf.complex(tf.zerosLike(P), P)));
    const wv = tf.signal.inverse_stft(
        SP,
        4 * hop,
        hop,
        {fftLength: 4 * hop, windowFn: tf.signal.inverseStftWindowFn(hop)}
    );

    return tf.squeeze(wv);
}
*/