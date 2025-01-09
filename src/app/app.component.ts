import {Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import * as tf from '@tensorflow/tfjs';


@Component({
    selector: 'app-root',
    standalone: true,
    templateUrl: './app.component.html',
    styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
    @ViewChild('canvas', { static: true }) canvas!: ElementRef<HTMLCanvasElement>;

    @ViewChild('waveform') waveformDiv!: ElementRef;
    @ViewChild('spectrogram') spectrogramDiv!: ElementRef;
    @ViewChild('audioPlayer') audioPlayer!: ElementRef<HTMLAudioElement>;

    sampleRate: number = 44100;

    audioBuffer!: AudioBuffer;
    frame_size = [1049344, 2];
    currentContinuousTensor: tf.Tensor = tf.zeros(this.frame_size);
    nextContinuousTensor: tf.Tensor = tf.zeros(this.frame_size);
    last_right_anchor: tf.Tensor = tf.ones([1, 128]);
    noiseg: tf.Tensor = tf.ones([1, 64]);
    genNoiseModel: any;
    genWaveformModel: any;
    isPlaying: boolean = false;
    audioContext!: AudioContext;
    sourceNode!: AudioBufferSourceNode | null;
    analyser!: AnalyserNode;
    animationFrameId!: number;
    blobs: Blob[] = [];
    currentBlobIndex: number = 0;
    canvasContext!: CanvasRenderingContext2D;
    currentBlob: Blob | null = null;

    async ngOnInit() {
        this.canvasContext = this.canvas.nativeElement.getContext('2d')!;
    }

    constructor() {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
    }

    addBlob(blob: Blob): void {
        this.blobs.push(blob);
    }

    async generatePlayback(): Promise<void> {
        console.log('generate playback');
        this.currentBlobIndex = 0;
        this.blobs = [];
        this.isPlaying = false;
        await this.audioContext.close()
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;

        await this.startContinuousGeneration();
        const waveBlob = await this.tensorToAudioBlob(this.currentContinuousTensor, this.sampleRate);
        this.addBlob(waveBlob);
        const nextWaveBlob = await this.tensorToAudioBlob(this.nextContinuousTensor, this.sampleRate);
        this.addBlob(nextWaveBlob);
        await this.startPlayback();
    }

    async startPlayback(): Promise<void> {
        if (this.isPlaying || this.blobs.length === 0) return;

        this.isPlaying = true;
        this.currentBlob = this.blobs[this.currentBlobIndex];

        const arrayBuffer = await this.currentBlob.arrayBuffer();
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

        this.sourceNode = this.audioContext.createBufferSource();
        this.sourceNode.buffer = audioBuffer;
        this.sourceNode.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);

        this.sourceNode.start(0);
        this.sourceNode.onended = async () => {
            this.isPlaying = false;
            this.currentBlobIndex = (this.currentBlobIndex + 1) % this.blobs.length;
            console.log('switch to new blob');
            await this.startPlayback();

            this.generateNextPart();
            const nextWaveBlob = await this.tensorToAudioBlob(this.nextContinuousTensor, this.sampleRate);
            this.addBlob(nextWaveBlob);
            console.log('new blob added');
        };

        this.visualize();
    }

    pausePlayback(): void {
        if (this.isPlaying) {
            this.audioContext.suspend();
            this.isPlaying = false;
            cancelAnimationFrame(this.animationFrameId);
        }
    }

    async resumePlayback(): Promise<void> {
        if (this.isPlaying || !this.currentBlob) return;
        await this.audioContext.resume();
        this.isPlaying = true;
        this.visualize();
        console.log('Audio resumed');
    }

    visualize(): void {
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const width = this.canvas.nativeElement.width;
        const height = this.canvas.nativeElement.height;

        const draw = () => {
            this.animationFrameId = requestAnimationFrame(draw);

            this.analyser.getByteFrequencyData(dataArray);

            this.canvasContext.fillStyle = '#000';
            this.canvasContext.fillRect(0, 0, width, height);

            const barWidth = (width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                this.canvasContext.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
                this.canvasContext.fillRect(x, height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        };

        draw();
    }

    async tensorToAudioBlob(tensor: tf.Tensor, sampleRate: number): Promise<Blob> {
        const audioData = await tensor.data() as Float32Array;

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const numberOfChannels = tensor.shape[1] || 1; // Handle mono or stereo
        const frameCount = audioData.length / numberOfChannels;

        this.audioBuffer = audioCtx.createBuffer(numberOfChannels, frameCount, sampleRate);

        for (let channel = 0; channel < numberOfChannels; channel++) {
            const channelData = this.audioBuffer.getChannelData(channel);
            for (let i = 0; i < frameCount; i++) {
                channelData[i] = audioData[i * numberOfChannels + channel];
            }
        }

        return this.audioBufferToWav(this.audioBuffer);
    }

    async startContinuousGeneration() {
        this.genNoiseModel = await tf.loadGraphModel('./assets/models/continous_noise_model_web/model.json');
        this.genWaveformModel = await tf.loadGraphModel('./assets/models/waveform_model_web/model.json');

        for (let i = 0; i < 3; i++) {
            const res = this.genNoiseModel.execute([this.noiseg, this.last_right_anchor]) as tf.Tensor[];
            const noise = res[1] as tf.Tensor;
            this.last_right_anchor = res[0] as tf.Tensor;
            console.log(noise.shape);
            console.log(this.last_right_anchor.shape);

            if (i > 0) {
                const predictions = this.genWaveformModel.execute(noise) as tf.Tensor;
                this.currentContinuousTensor = this.nextContinuousTensor
                this.nextContinuousTensor = predictions;
            }
        }
    }

    generateNextPart() {
        const res = this.genNoiseModel.execute([this.noiseg, this.last_right_anchor]) as tf.Tensor[];
        console.log('res', res);
        const noise = res[1] as tf.Tensor;
        this.last_right_anchor = res[0] as tf.Tensor;
        console.log(noise.shape);
        console.log(this.last_right_anchor.shape);
        const predictions = this.genWaveformModel.execute(noise) as tf.Tensor;
        console.log('Predictions:', predictions);
        console.log('res', predictions.shape);
        this.currentContinuousTensor = this.nextContinuousTensor;
        this.nextContinuousTensor = predictions;
    }

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

        function setUint16(data: number) {
            view.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data: number) {
            view.setUint32(pos, data, true);
            pos += 4;
        }

        setUint32(0x46464952);
        setUint32(length - 8);
        setUint32(0x45564157);

        setUint32(0x20746d66);
        setUint32(16);
        setUint16(1);
        setUint16(numOfChan);
        setUint32(sampleRate);
        setUint32(sampleRate * numOfChan * bitDepth / 8);
        setUint16(numOfChan * bitDepth / 8);
        setUint16(bitDepth);

        setUint32(0x61746164);
        setUint32(length - pos - 4);

        for (let i = 0; i < numOfChan; i++) {
            channels.push(buffer.getChannelData(i));
        }
        while (pos < length) {
            for (let i = 0; i < numOfChan; i++) {
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