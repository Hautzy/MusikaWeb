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

    sampleRate: number = 44100;
    isGenerating: boolean = false;
    frameDurationSeconds = 23.79465;

    audioBuffer!: AudioBuffer;
    frameSize = [1049344, 2];
    
    last_right_anchor: tf.Tensor = tf.ones([1, 128]);
    noiseg: tf.Tensor = tf.ones([1, 64]);
    genNoiseModel: any;
    genWaveformModel: any;
    isPlaying: boolean = false;
    audioContext!: AudioContext;
    analyser!: AnalyserNode;
    animationFrameId!: number;
    canvasContext!: CanvasRenderingContext2D;

    currentContinuousTensor: tf.Tensor = tf.zeros(this.frameSize);
    nextContinuousTensor: tf.Tensor = tf.zeros(this.frameSize);
    maxMusicFrame: number = 0;

    currentBuffer: AudioBuffer | null = null;
    nextBuffer: AudioBuffer | null = null;

    sourceNodeOne!: AudioBufferSourceNode | null;
    sourceNodeTwo!: AudioBufferSourceNode | null;

    async ngOnInit(): Promise<void> {
        this.genNoiseModel = await tf.loadGraphModel('./assets/models/continous_noise_model_web/model.json');
        this.genWaveformModel = await tf.loadGraphModel('./assets/models/waveform_model_web/model.json');
        console.log('models loaded');
        this.canvasContext = this.canvas.nativeElement.getContext('2d')!;
    }

    constructor() {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
    }

    async generatePlayback(): Promise<void> {
        this.isGenerating = true;
        if (this.sourceNodeOne && this.sourceNodeTwo) {
            cancelAnimationFrame(this.animationFrameId);
            this.sourceNodeOne.stop();
            this.sourceNodeTwo.stop();
            this.maxMusicFrame = 0;
            this.isPlaying = false;
        }
        console.log('generate playback');

        await this.startContinuousGeneration();
        this.currentBuffer = await this.tensorToAudioBuffer(this.currentContinuousTensor, this.sampleRate);
        await this.startPlayback(this.currentBuffer);
        this.maxMusicFrame++;
        this.nextBuffer = await this.tensorToAudioBuffer(this.nextContinuousTensor, this.sampleRate);
        await this.startPlayback(this.nextBuffer);
        this.maxMusicFrame++;
        this.isGenerating = false;
    }

    async startPlayback(buffer: AudioBuffer): Promise<void> {
        if (!buffer) return;

        this.isPlaying = true;


        const sourceNode = this.audioContext.createBufferSource();

        if (this.maxMusicFrame % 2 == 0) {
            this.sourceNodeOne = sourceNode;
            console.log('take source node ONE!')
        } else {
            this.sourceNodeTwo = sourceNode;
            console.log('take source node TWO!')
        }

        sourceNode.buffer = buffer;
        sourceNode.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);

        console.log(`schedule ${this.maxMusicFrame+1} blob`)
        console.log(`start at ${this.maxMusicFrame * this.frameDurationSeconds} secs`)

        sourceNode.start(this.maxMusicFrame * this.frameDurationSeconds);

        sourceNode.onended = async (): Promise<void> => {
            if (this.isGenerating) {
                return;
            }
            this.isPlaying = false;
            console.log(`generate ${this.maxMusicFrame+1} blob`);
            this.generateNextPart();
            this.currentBuffer = this.nextBuffer;
            this.nextBuffer = await this.tensorToAudioBuffer(this.nextContinuousTensor, this.sampleRate);
            await this.startPlayback(this.nextBuffer);
            this.maxMusicFrame++;
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
        if (this.isPlaying || !this.currentBuffer) return;
        await this.audioContext.resume();
        this.isPlaying = true;
        this.visualize();
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

    async startContinuousGeneration() {
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
        const noise = res[1] as tf.Tensor;
        this.last_right_anchor = res[0] as tf.Tensor;
        const predictions = this.genWaveformModel.execute(noise) as tf.Tensor;
        this.currentContinuousTensor = this.nextContinuousTensor;
        this.nextContinuousTensor = predictions;
    }

    async tensorToAudioBuffer(tensor: tf.Tensor, sampleRate: number): Promise<AudioBuffer> {
        const audioData = await tensor.data() as Float32Array;
        const numberOfChannels = tensor.shape[1] || 1; // Mono oder Stereo
        const frameCount = audioData.length / numberOfChannels;

        const audioBuffer = this.audioContext.createBuffer(numberOfChannels, frameCount, sampleRate);

        for (let channel = 0; channel < numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            const offset = channel;
            channelData.set(audioData.filter((_, i) => i % numberOfChannels === offset));
        }

        return audioBuffer;
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