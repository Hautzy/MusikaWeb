import {Component, ElementRef, OnDestroy, ViewChild} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import WaveSurfer from 'wavesurfer.js';
import SpectrogramPlugin from 'wavesurfer.js/dist/plugins/spectrogram.js';


@Component({
    selector: 'app-root',
    standalone: true,
    templateUrl: './app.component.html',
    styleUrl: './app.component.css'
})
export class AppComponent implements OnDestroy{

    @ViewChild('waveform') waveformDiv!: ElementRef;
    @ViewChild('spectrogram') spectrogramDiv!: ElementRef;
    @ViewChild('audioPlayer') audioPlayer!: ElementRef<HTMLAudioElement>;

    tensorData: number[][] = [];
    sampleRate: number = 44100; // Default to 44.1 kHz sample rate for audio

    private waveSurfer!: WaveSurfer;
    private wavBlob!: Blob;
    private audioBuffer!: AudioBuffer;

    zoomLevel: number = 20; // Initial zoom level
    isPlaying: boolean = false;


    ngOnDestroy() {
        if (this.waveSurfer) {
            this.waveSurfer.destroy();
        }
    }

    async startStereo(): Promise<void> {
        const model = await tf.loadGraphModel('./assets/models/stereo_model_web/model.json');
        const inputTensor = tf.zeros([1, 1]);  // Generate or change input as needed
        const predictions = await model.executeAsync(inputTensor) as tf.Tensor;
        console.log('res', predictions.shape);

        // Store the tensor data for WAV creation
        this.tensorData = await predictions.array() as number[][];

        // Convert the tensor to an audio Blob
        this.wavBlob = await this.tensorToAudioBlob(predictions, this.sampleRate);

        // Create a Blob URL
        const blobUrl = URL.createObjectURL(this.wavBlob);

        // Initialize WaveSurfer.js
        if (this.waveSurfer) {
            this.waveSurfer.destroy();
        }
        this.waveSurfer = WaveSurfer.create({
            container: this.waveformDiv.nativeElement,
            waveColor: 'violet',
            progressColor: 'purple',
            normalize: true,
            height: 200,
            minPxPerSec: this.zoomLevel, // Controls zoom level and enables scrolling
            autoCenter: true,            // Auto-centers the waveform during playback
            plugins: [
                SpectrogramPlugin.create({
                    container: this.spectrogramDiv.nativeElement,
                    labels: true,
                }),
            ],
        });


        // Load the audio Blob URL into WaveSurfer
        this.waveSurfer.load(blobUrl);

        // Update playback state
        this.waveSurfer.on('play', () => {
            this.isPlaying = true;
        });

        this.waveSurfer.on('pause', () => {
            this.isPlaying = false;
        });

        this.waveSurfer.on('finish', () => {
            this.isPlaying = false;
            this.waveSurfer.seekTo(0); // Reset to start
        });

        // Optional: Play the audio using the native audio element
        this.playWavInBrowser();
        console.log('finished');
    }

    /*
    * async runInferenceWaveform(): Promise<void> {
    const model = await tf.loadGraphModel('./assets/models/waveform_model_web/model.json');
    const seconds = 120
    const fac = Math.ceil(seconds / 23) + 1
    let input = getNoiseInterpMulti(fac);
    console.log('noise input', input);

    const predictions = model.execute(input) as tf.Tensor[];
    console.log('Predictions:', predictions);

    const S = predictions[0];
    const P = predictions[1];

    console.log('S:', S);
    console.log('P:', P);


    const result = await this.computeISTFT(S, P);
    console.log('result:', result);

    // Squeeze the tensor
    const squeezedTensor = result.squeeze();  // Shape: [4096, 2] (no change in this case)
    console.log('squeezedTensor:', squeezedTensor);

    // Clip the values to [-1.0, 1.0]
    const clippedTensor = squeezedTensor.clipByValue(-1.0, 1.0);  // Shape: [4096, 2]
    console.log('clippedTensor:', clippedTensor);
    await this.visualizeWaveform(result);
  }
    * */

    playPause() {
        if (this.waveSurfer) {
            this.waveSurfer.playPause();
            // isPlaying state will be updated via event listeners
        }
    }

    stop() {
        if (this.waveSurfer) {
            this.waveSurfer.stop();
            this.isPlaying = false;
        }
    }

    playWavInBrowser(): void {
        if (!this.wavBlob) {
            console.error('Audio Blob is not available.');
            return;
        }
        const url = window.URL.createObjectURL(this.wavBlob);

        // Set the audio player's source to the generated WAV file
        const audioElement = this.audioPlayer.nativeElement;
        audioElement.src = url;
        audioElement.load();
    }

    downloadTensorAsJson() {
        const jsonData = JSON.stringify(this.tensorData);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'tensor.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    downloadWav(): void {
        if (!this.wavBlob) {
            console.error('WAV Blob is not available.');
            return;
        }
        const url = window.URL.createObjectURL(this.wavBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'audio.wav';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    // Convert tensor to audio Blob
    async tensorToAudioBlob(tensor: tf.Tensor, sampleRate: number): Promise<Blob> {
        const audioData = await tensor.data() as Float32Array;

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const numberOfChannels = tensor.shape[1] || 1; // Handle mono or stereo
        const frameCount = audioData.length / numberOfChannels;

        this.audioBuffer = audioCtx.createBuffer(numberOfChannels, frameCount, sampleRate);

        // Fill the buffer with the data
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const channelData = this.audioBuffer.getChannelData(channel);
            for (let i = 0; i < frameCount; i++) {
                channelData[i] = audioData[i * numberOfChannels + channel];
            }
        }

        // Convert AudioBuffer to WAV Blob
        return this.audioBufferToWav(this.audioBuffer);
    }

    // Function to convert AudioBuffer to WAV Blob
    audioBufferToWav(buffer: AudioBuffer): Blob {
        const numOfChan = buffer.numberOfChannels,
            length = buffer.length * numOfChan * 2 + 44,
            bufferArray = new ArrayBuffer(length),
            view = new DataView(bufferArray),
            channels = [],
            sampleRate = buffer.sampleRate,
            bitDepth = 16;
        let offset = 0,
            pos = 0;

        // Write WAV header
        function setUint16(data: number) {
            view.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data: number) {
            view.setUint32(pos, data, true);
            pos += 4;
        }

        // RIFF chunk descriptor
        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8); // File size - 8
        setUint32(0x45564157); // "WAVE"

        // fmt sub-chunk
        setUint32(0x20746d66); // "fmt "
        setUint32(16); // Subchunk1Size (16 for PCM)
        setUint16(1); // AudioFormat (1 for PCM)
        setUint16(numOfChan);
        setUint32(sampleRate);
        setUint32(sampleRate * numOfChan * bitDepth / 8); // Byte rate
        setUint16(numOfChan * bitDepth / 8); // Block align
        setUint16(bitDepth);

        // data sub-chunk
        setUint32(0x61746164); // "data"
        setUint32(length - pos - 4); // Subchunk2Size

        // Write interleaved PCM samples
        for (let i = 0; i < numOfChan; i++) {
            channels.push(buffer.getChannelData(i));
        }

        const sampleCount = buffer.length;
        while (pos < length) {
            for (let i = 0; i < numOfChan; i++) {
                // Interleave channels
                let sample = Math.max(-1, Math.min(1, channels[i][offset]));
                sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                view.setInt16(pos, sample, true);
                pos += 2;
            }
            offset++;
        }

        return new Blob([bufferArray], { type: 'audio/wav' });
    }
}