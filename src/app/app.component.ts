import {Component, OnInit} from '@angular/core';
import { RouterOutlet } from '@angular/router';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {zip} from "rxjs";
import {HOP} from "../../functions/src/constant";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  async ngOnInit(): Promise<void> {
    await this.runInferenceStereo();
  }

  async testModels(): Promise<void> {
    console.log('Load dec_model');
    const dec_model = await tf.loadGraphModel('./assets/models/dec_model_web/model.json');
    console.log('Load dec2_model');
    const dec2_model = await tf.loadGraphModel('./assets/models/dec2_model_web/model.json');
    console.log('Load gen_ema_model');
    const gen_ema_model = await tf.loadGraphModel('./assets/models/gen_ema_model_web/model.json');
    console.log('Load stereo_model');
    const stereo_model = await tf.loadGraphModel('./assets/models/stereo_model_web/model.json');

    const model_names = ['dec_model', 'dec2_model', 'gen_ema_model', 'stereo_model'];
    const models = [dec_model, dec2_model, gen_ema_model, stereo_model];

    console.log('Models loaded');
    await zip(model_names, models).forEach(([name, model]) => {
      console.log(name);
      console.log('Inputs and outputs:');
      console.log(model.inputs);
      console.log(model.outputs);
    });

    console.log('dec');
    console.log('Predictions:');
    console.log(dec_model.predict(tf.randomNormal([2, 1, 64, 64])));

    console.log('dec2');
    console.log('Predictions:');
    console.log(dec2_model.predict(tf.randomNormal([2, 1, 4, 64])));

    console.log('gen_ema');
    console.log('Predictions:');
    console.log(gen_ema_model.predict(tf.randomNormal([2, 256, 128])));

    console.log('stereo');
    console.log('Predictions:');
    const inputTensor = tf.zeros([1, 1]);  // Change the shape as needed
    const predictions = await stereo_model.executeAsync(inputTensor) as tf.Tensor[];

    // Type assertions to access S and P
    const S: tf.Tensor = predictions[0];
    const P: tf.Tensor = predictions[1];

    // Log the tensors
    console.log('S:', S);
    S.print();
    console.log('P:', P);
    P.print();
  }

  hannWindow(length: number): tf.Tensor {
    const window = [];
    for (let i = 0; i < length; i++) {
      window.push(0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1))));
    }
    return tf.tensor(window);
  }

  // Function to compute inverse short-time Fourier transform (ISTFT)
  async computeISTFT(S: tf.Tensor, P: tf.Tensor) {
    // Convert S and P to tensors
    const fftLength = 4 * HOP;
    const hop = HOP;

    // Perform element-wise operations to get complex spectrum
    const expP = tf.complex(tf.cos(P), tf.sin(P));
    console.log('expP:', expP);
    const SP = tf.mul(tf.complex(S, tf.zerosLike(S)), expP);
    console.log('SP:', SP);

    // Perform inverse FFT
    const wvFrames = tf.irfft(SP);
    console.log('wvFrames:', wvFrames);

    // Apply the window function
    const window = this.hannWindow(fftLength);
    console.log('window:', window);

    // Reshape and tile the window to match the dimensions of wvFrames
    const reshapedWindow = window.reshape([fftLength, 1, 1]);
    console.log('reshapedWindow:', reshapedWindow);
    const tiledWindow = reshapedWindow.tile([Math.ceil(wvFrames.shape[0] / fftLength), wvFrames.shape[1]!, wvFrames.shape[2]!]);
    console.log('tiledWindow:', tiledWindow);
    // Slice the tiled window to exactly match wvFrames shape
    const windowedWvFrames = wvFrames.mul(tiledWindow.slice([0, 0, 0], wvFrames.shape));
    console.log('windowedWvFrames:', windowedWvFrames);

    // Overlap and add (OLA) to reconstruct the signal
    const frameStep = hop;  // Hop size
    const frameLength = fftLength;  // Frame length
    console.log('frameStep:', frameStep);
    const numFrames = Math.floor((wvFrames.shape[0] - fftLength) / frameStep) + 1;  // Number of frames
    console.log('numFrames:', numFrames);
    const outputLength = numFrames * frameStep + fftLength - frameStep;  // Length of the reconstructed signal
    console.log('outputLength:', outputLength);
    const reconstructed = tf.buffer([outputLength, 2], 'float32');  // Shape: [1049344, 2]
    console.log('reconstructed:', reconstructed);

    for (let i = 0; i < numFrames; i++) {
      const start = i * frameStep;
      const frame = windowedWvFrames.slice([i * frameStep], [frameLength]);  // Shape: [1024, 513, 2]
      console.log('frame:', frame);
      // Add the frame to the reconstructed signal
      const frameArray = frame.arraySync() as number[][][];  // Convert to array for setting values
      for (let j = 0; j < frameLength; j++) {
        for (let k = 0; k < 2; k++) {  // Loop over the channels
          const currentValue = reconstructed.get(start + j, k) as number;  // Get current value for channel
          reconstructed.set(currentValue + frameArray[j][0][k], start + j, k);  // Add value to reconstructed signal for channel
        }
      }
    }

    console.log('reconstructed:', reconstructed);
    return reconstructed.toTensor();
  }

  async runInferenceStereo(): Promise<void> {
    const stereo_model = await tf.loadGraphModel('./assets/models/stereo_model_web/model.json');
    const inputTensor = tf.zeros([1, 1]);  // Change the shape as needed
    const predictions = await stereo_model.executeAsync(inputTensor) as tf.Tensor[];

    const S = predictions[0];
    const P = predictions[1];

    console.log('S:', S);
    console.log('P:', P);

    const result = await this.computeISTFT(S, P);
    await this.visualizeWaveform(result);
  }

  async visualizeWaveform(reconstructed: tf.Tensor) {
    // Convert the tensor to a regular array
    const reconstructedArray: number[][] = await reconstructed.array() as number[][];

    // Extract the left and right channels
    const leftChannel = reconstructedArray.map(sample => sample[0]);
    const rightChannel = reconstructedArray.map(sample => sample[1]);

    // Prepare the data for tfvis
    const leftChannelData = leftChannel.map((value, index) => ({
      x: index,
      y: value
    }));

    const rightChannelData = rightChannel.map((value, index) => ({
      x: index,
      y: value
    }));

    // Visualize the left channel
    tfvis.render.linechart(
        { name: 'Left Channel Waveform', tab: 'Charts' },
        { values: [leftChannelData], series: ['Left Channel'] },
        {
          xLabel: 'Sample',
          yLabel: 'Amplitude',
          width: 800,
          height: 400,
        }
    );

    // Visualize the right channel
    tfvis.render.linechart(
        { name: 'Right Channel Waveform', tab: 'Charts' },
        { values: [rightChannelData], series: ['Right Channel'] },
        {
          xLabel: 'Sample',
          yLabel: 'Amplitude',
          width: 800,
          height: 400,
        }
    );
  }
}