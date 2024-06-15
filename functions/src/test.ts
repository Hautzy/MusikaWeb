import * as tf from '@tensorflow/tfjs';
import { writeFileSync } from 'fs';
import {centerCoordinate} from "./utils";

const testTensor = tf.tensor3d([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]);
const result = centerCoordinate(testTensor);

result.array().then(array => {
    writeFileSync('./test_output/output.json', JSON.stringify(array));
    console.log('Result written to output.json');
});