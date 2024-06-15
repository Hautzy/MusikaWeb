import * as tf from '@tensorflow/tfjs';
import { writeFileSync } from 'fs';
import {centerCoordinate, truncatedNormal, interpolate, getNoiseInterpMulti} from "./utils";


export function testCenterCoordinate(): void {
    const testTensor = tf.tensor3d([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]);
    const result = centerCoordinate(testTensor);

    result.array().then(array => {
        writeFileSync('./test_output/testCenterCoordinate.json', JSON.stringify(array));
    });
}

export function testTruncatedNormal(): void {
    const shape = [9, 9];
    const bound = 2.0;
    const result = truncatedNormal(shape, bound);

    result.array().then(array => {
        writeFileSync('./test_output/testTruncatedNormal.json', JSON.stringify(array));
    });
}

export function testInterpolate(): void {
    const start = tf.tensor2d([[2, 2], [2, 2]]);
    const end = tf.tensor2d([[6, 6], [6, 6]]);
    //console.log(interpolate(start, end, 3));
    console.log(getNoiseInterpMulti());
}

//testCenterCoordinate();
//testTruncatedNormal();
testInterpolate()