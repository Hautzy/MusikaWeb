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

    frameSize = [1049344, 2];
    
    last_right_anchor!: tf.Tensor;
    noiseg!: tf.Tensor;
    genNoiseModel: any;
    genWaveformModel: any;
    isPlaying: boolean = false;
    audioContext!: AudioContext;
    analyser!: AnalyserNode;
    animationFrameId!: number;
    canvasContext!: CanvasRenderingContext2D;

    currentBuffer: AudioBuffer | null = null;
    nextBuffer: AudioBuffer | null = null;

    currentSourceNode!: AudioBufferSourceNode | null;
    nextSourceNode!: AudioBufferSourceNode | null;

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
        this.analyser.connect(this.audioContext.destination);
    }

    async generatePlayback(): Promise<void> {
        this.isGenerating = true;
        if (this.currentSourceNode) {
            console.log('stop current source node');
            cancelAnimationFrame(this.animationFrameId);
            this.currentSourceNode.stop();
            this.nextStartTime = -1.0;
            this.isPlaying = false;
        }
        if (this.nextSourceNode) {
            this.nextSourceNode.stop();
        }
        console.log('generate playback');

        await this.resetNoise();
        this.currentBuffer = await this.generateNextPart();
        this.currentSourceNode = await this.schedulePlayback(this.currentBuffer);
        this.currentSourceNode.connect(this.analyser);
        this.visualize();
        this.nextBuffer = await this.generateNextPart();
        this.nextSourceNode = await this.schedulePlayback(this.nextBuffer);
        this.isGenerating = false;
    }

    nextStartTime: number = -1.0;

    async schedulePlayback(buffer: AudioBuffer): Promise<AudioBufferSourceNode> {
        this.isPlaying = true;

        const sourceNode = this.audioContext.createBufferSource();
        sourceNode.buffer = buffer;

        if(this.nextStartTime < 0.0) {
            this.nextStartTime = this.audioContext.currentTime + 0.1;
        }
        sourceNode.start(this.nextStartTime);
        console.log(`scheduled playback of ${buffer.duration}s at: ${this.nextStartTime}`);
        this.nextStartTime += buffer.duration;
        console.log(`next start time: ${this.nextStartTime}`);

        sourceNode.onended = async (): Promise<void> => { this.chunkEnded() };

        return sourceNode;
    }

    async chunkEnded(): Promise<void> {
        this.currentBuffer = this.nextBuffer;
        this.currentSourceNode = this.nextSourceNode;
        if (this.currentSourceNode != null) {
            this.currentSourceNode.connect(this.analyser);
        }
        if (this.isGenerating) {
            return;
        }
        this.isGenerating = true;
        this.nextBuffer = await this.generateNextPart();
        this.isGenerating = false;
        this.nextSourceNode = await this.schedulePlayback(this.nextBuffer);
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

            const barWidth = (width / bufferLength);
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i];
                this.canvasContext.fillStyle = `rgb(85, 85, ${barHeight / 2 + 100})`;
                this.canvasContext.fillRect(x, height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        };

        draw();
    }

    async resetNoise() {
        this.noiseg = tf.truncatedNormal([1, 64]);
        console.log('Initialized global noise tensors');
        this.last_right_anchor = tf.truncatedNormal([1, 128]);
        console.log('Initialized first anchor noise tensors');
    }

    async generateNextPart(): Promise<AudioBuffer> {
        console.log('generating another chunk...');
        const res = this.genNoiseModel.execute([this.noiseg, this.last_right_anchor]) as tf.Tensor[];
        const noise = res[1] as tf.Tensor;
        this.last_right_anchor = res[0] as tf.Tensor;
        const predictions = await this.genWaveformModel.execute(noise) as tf.Tensor;
        console.log(`generated ${predictions.shape} samples.`);
        return this.tensorToAudioBuffer(predictions, this.sampleRate);
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

    playButtonClicked() {
        if (this.noiseg) {
            this.resumePlayback();
        } else {
            this.generatePlayback();
        }
    }
}
