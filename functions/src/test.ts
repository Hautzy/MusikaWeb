import * as tf from '@tensorflow/tfjs';
import { writeFileSync } from 'fs';
import {centerCoordinate, generateLinspace, getNoiseInterpMulti, truncatedNormal} from "./utils";


function testCenterCoordinate(): void {
    const testTensor = tf.tensor3d([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]);
    const result = centerCoordinate(testTensor);

    result.array().then(array => {
        writeFileSync('./test_output/testCenterCoordinate.json', JSON.stringify(array));
    });
}

function testTruncatedNormal(): void {
    const shape = [9, 9];
    const bound = 2.0;
    const result = truncatedNormal(shape, bound);

    result.array().then(array => {
        writeFileSync('./test_output/testTruncatedNormal.json', JSON.stringify(array));
    });
}

function testLinespaceInterpolateAtAxis(): void {
    const startTensor = tf.tensor([[0.0, 1.0], [2.0, 3.0]]);  // Shape (2, 2)
    const endTensor = tf.tensor([[10.0, 11.0], [12.0, 13.0]]);  // Shape (2, 2)
    const steps = 5;

    const result = generateLinspace(startTensor, endTensor, steps);
    console.log(result.shape);
}

function testNoiseInterpMulti(): void {
    const result = getNoiseInterpMulti();

    console.assert(result.shape[0] === 1);
    console.assert(result.shape[1] === 128);
    console.assert(result.shape[2] === 128);
    console.log(result.shape)
    console.log(result);
}

//testCenterCoordinate();
//testTruncatedNormal();
//testLinespaceInterpolateAtAxis()
testNoiseInterpMulti();
