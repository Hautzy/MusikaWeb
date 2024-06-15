import * as tf from '@tensorflow/tfjs';
import {db2power, denormalize} from "./generation";

function testDenormalize() {
    const S = tf.tensor([0.5, -0.5, 0.0]);
    const result = denormalize(S);
    result.print()
}

function testDb2Power() {
    const S_db = tf.tensor([2.0, 4.0, 0.0]);
    const result = db2power(S_db);
    result.print()
}

testDenormalize();
testDb2Power();