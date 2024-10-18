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
    noise: tf.Tensor | null = null;

    private waveSurfer!: WaveSurfer;
    private wavBlob!: Blob;
    private audioBuffer!: AudioBuffer;

    zoomLevel: number = 20; // Initial zoom level
    isPlaying: boolean = false;

    selectFile() {
        const fileInput = document.querySelector('input[type="file"]') as HTMLElement;
        fileInput.click();
    }

    onFileSelected(event: any) {
        const file: File = event.target.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = async (e: any) => {
                const jsonData = JSON.parse(e.target.result);
                this.loadTensorFromJSON(jsonData);
            };

            reader.readAsText(file);
        }
    }

    loadTensorFromJSON(jsonData: any) {
        const { data, shape } = jsonData;
        this.noise = tf.tensor(data, shape);
        console.log('Tensor loaded:', this.noise);
        this.noise.print(); // To display the tensor in the console
    }

    ngOnDestroy() {
        if (this.waveSurfer) {
            this.waveSurfer.destroy();
        }
    }

    async startInferenceStereo(): Promise<void> {
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

        //this.playWavInBrowser();
        console.log('finished');
    }

    async startInferenceWaveform(): Promise<void> {
        const noise_model = await tf.loadGraphModel('./assets/models/noise_model_web/model.json');
        const model = await tf.loadGraphModel('./assets/models/waveform_model_web_fac_6/model.json');
        const seconds = 120
        const fac = Math.floor(seconds / 23) + 1
        console.log('fac', fac);

        let noise_predictions = this.noise;
        if (noise_predictions == null) {
            console.log('No noise tensor loaded, generating one...');
            // TODO: make input of noise prediction through input
            const inputTensor = tf.zeros([1, 1]);  // Generate or change input as needed
            noise_predictions = await noise_model.executeAsync(inputTensor) as tf.Tensor;
        }
        console.log('noise_predictions', noise_predictions.shape);
        console.log('noise_predictions', noise_predictions);

        const predictions = await model.executeAsync(noise_predictions) as tf.Tensor;
        console.log('Predictions:', predictions);
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

        //this.playWavInBrowser();
        console.log('finished');
  }

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

    /*playWavInBrowser(): void {
        if (!this.wavBlob) {
            console.error('Audio Blob is not available.');
            return;
        }
        const url = window.URL.createObjectURL(this.wavBlob);

        // Set the audio player's source to the generated WAV file
        const audioElement = this.audioPlayer.nativeElement;
        audioElement.src = url;
        audioElement.load();
    }*/

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

    frame_size = [1049344, 2];
    currentContinuousTensor: tf.Tensor = tf.zeros(this.frame_size);
    nextContinuousTensor: tf.Tensor = tf.zeros(this.frame_size);

    last_right_anchor: tf.Tensor = tf.ones([1, 128]);

    noiseg: tf.Tensor = tf.ones([1, 64]);

    genNoiseModel: any;
    genWaveformModel: any;

    async startContinuousGeneration() {
        this.genNoiseModel = await tf.loadGraphModel('./assets/models/continous_noise_model_web/model.json');
        this.genWaveformModel = await tf.loadGraphModel('./assets/models/waveform_model_web/model.json');

        for (let i = 0; i < 3; i++) {
            console.log('i', i);
            const res = this.genNoiseModel.execute([this.noiseg, this.last_right_anchor]) as tf.Tensor[];
            console.log('res', res);
            const noise = res[0] as tf.Tensor;
            this.last_right_anchor = res[1] as tf.Tensor;
            console.log(noise.shape);
            console.log(this.last_right_anchor.shape);

            if (i > 0) {
                const predictions = this.genWaveformModel.execute(noise) as tf.Tensor;
                console.log('Predictions:', predictions);
                console.log('res', predictions.shape);
                this.currentContinuousTensor = this.nextContinuousTensor
                this.nextContinuousTensor = predictions;
            }
        }
        await this.startPlaying(this.currentContinuousTensor)
        // start playing
        // handler if current finished generate new and in parallel start music generation
    }

    async startPlaying(toPlayAudioTensor: tf.Tensor) {
        // Convert the tensor to an audio Blob
        this.wavBlob = await this.tensorToAudioBlob(toPlayAudioTensor, this.sampleRate);

        // Create a Blob URL
        const blobUrl = URL.createObjectURL(this.wavBlob);

        // Check if WaveSurfer is already initialized
        if (!this.waveSurfer) {
            // Initialize WaveSurfer.js if not already created
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

            // Update playback state
            this.waveSurfer.on('play', () => {
                this.isPlaying = true;
            });

            this.waveSurfer.on('pause', () => {
                this.isPlaying = false;
            });

            this.waveSurfer.on('finish', async () => {
                await this.startPlaying(this.nextContinuousTensor);
                this.waveSurfer.seekTo(0); // Reset to start
                await this.waveSurfer.play();
                this.generateNextPart();
            });
        }

        // Load the new audio Blob URL without destroying the WaveSurfer instance
        this.waveSurfer.load(blobUrl);

        // Play the audio immediately after loading
        this.waveSurfer.on('ready', async () => {
            await this.waveSurfer.play();
        });

        console.log('finished');
    }


    generateNextPart() {
        const res = this.genNoiseModel.execute([this.noiseg, this.last_right_anchor]) as tf.Tensor[];
        console.log('res', res);
        const noise = res[0] as tf.Tensor;
        this.last_right_anchor = res[1] as tf.Tensor;
        console.log(noise.shape);
        console.log(this.last_right_anchor.shape);
        const predictions = this.genWaveformModel.execute(noise) as tf.Tensor;
        console.log('Predictions:', predictions);
        console.log('res', predictions.shape);
        this.currentContinuousTensor = this.nextContinuousTensor;
        this.nextContinuousTensor = predictions;
    }
}