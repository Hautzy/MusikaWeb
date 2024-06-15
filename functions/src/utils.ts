import * as tf from '@tensorflow/tfjs';

export const COOR_DEPTH = 64;
export const LAT_LEN = 128;
export const COOR_LEN = Math.floor(LAT_LEN / 2) * 3;

export function centerCoordinate(x: tf.Tensor): tf.Tensor {
    const shape = x.shape;
    const shift = -1;
    const axis = -2;

    // Calculate the dimension along which to roll
    const axisDim = shape[shape.length + axis];

    // Roll the tensor
    const rolled = tf.concat([x.slice([0, -shift], [-1, axisDim + shift]), x.slice([0, 0], [-1, -shift])], axis);

    // Stack the original and rolled tensors
    const stacked = tf.stack([x, rolled], 0);

    // Compute the mean
    const mean = tf.mean(stacked, 0);

    // Slice the mean tensor to exclude the last element of the second dimension
    return mean.slice([0, 0, 0], [-1, shape[shape.length - 2] - 1, shape[shape.length - 1]]);

}

export function truncatedNormal(shape: number[], bound: number = 2.0): tf.Tensor {
    const seed = Math.floor(Math.random() * (2 ** 31));
    const mean = 0.0;
    const stddev = 1.0;

    // Generate the truncated normal distribution tensor
    const tensor = tf.tidy(() => {
        let tensor = tf.randomNormal(shape, mean, stddev, 'float32', seed);
        tensor = tensor.where(tensor.greaterEqual(tf.scalar(-bound)), tf.scalar(-bound));
        tensor = tensor.where(tensor.lessEqual(tf.scalar(bound)), tf.scalar(bound));
        return tensor;
    });

    return tensor;
}

export function generateLinspace(start: tf.Tensor, end: tf.Tensor, steps: number): tf.Tensor {
    const startExp = start.expandDims(1);
    const endExp = end.expandDims(1);
    const linspaceVals = tf.linspace(0, 1, steps).reshape([1, steps, 1]);

    return startExp.mul(tf.scalar(1).sub(linspaceVals)).add(endExp.mul(linspaceVals));
}

export function getNoiseList(fac: number, variance: number, noiseg: tf.Tensor): tf.Tensor[] {
    const coordratio = Math.floor(COOR_LEN / LAT_LEN);
    const noiselsLen = Array.from({ length: 3 + Math.floor((fac - 1) / coordratio) }, (_, i) => i);

    return noiselsLen.map(() =>
        tf.concat([truncatedNormal([1, 64], variance), noiseg], -1)
    );
}

export function getRls(noisels: tf.Tensor[]): tf.Tensor {
    const rlsLen = Array.from({ length: noisels.length - 1 }, (_, i) => i);
    const rlsParts = rlsLen.map(k =>
        generateLinspace(noisels[k], noisels[k + 1], COOR_LEN + 1).slice([0, 0, 0], [-1, -1, -1])
    );

    return tf.concat(rlsParts, -2);
}

export function processRls(rls: tf.Tensor): tf.Tensor {
    let processedRls = centerCoordinate(rls);
    processedRls = processedRls.slice([0, Math.floor(LAT_LEN / 4), 0], [-1, -1, -1]);
    processedRls = processedRls.slice([0, 0, 0], [-1, Math.floor(processedRls.shape[1]! / LAT_LEN) * LAT_LEN, -1]);

    return processedRls;
}

export function getNoiseInterpMulti(fac: number = 1, variance: number = 2.0): tf.Tensor {
    const noiseg = truncatedNormal([1, COOR_DEPTH], variance);

    const noisels = getNoiseList(fac, variance, noiseg);
    let rls = getRls(noisels);
    rls = processRls(rls);

    const splitRls = tf.split(rls, rls.shape[1]! / LAT_LEN, 1);

    return tf.concat(splitRls.slice(0, fac), 0);
}