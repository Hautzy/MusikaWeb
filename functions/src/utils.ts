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