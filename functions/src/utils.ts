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