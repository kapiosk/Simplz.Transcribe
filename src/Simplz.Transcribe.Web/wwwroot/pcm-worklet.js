// AudioWorkletProcessor: resamples the device rate (the worklet-global
// `sampleRate`) down to 16 kHz and posts Int16 PCM buffers to the main thread.
const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 2048; // 128 ms at 16 kHz per posted buffer

class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.readPos = 0;          // fractional read position into `tail` + current input
    this.tail = new Float32Array(0);
    this.out = new Int16Array(CHUNK_SAMPLES);
    this.outLen = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    // Work on tail-of-previous + current block so interpolation can cross block edges.
    const input = new Float32Array(this.tail.length + channel.length);
    input.set(this.tail, 0);
    input.set(channel, this.tail.length);

    let pos = this.readPos;
    while (pos + 1 < input.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const sample = input[i] * (1 - frac) + input[i + 1] * frac; // linear interpolation
      const clamped = Math.max(-1, Math.min(1, sample));
      this.out[this.outLen++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      if (this.outLen === CHUNK_SAMPLES) {
        const buf = this.out.slice();
        this.port.postMessage(buf, [buf.buffer]);
        this.outLen = 0;
      }
      pos += this.ratio;
    }

    // Keep the last sample for cross-block interpolation.
    const keepFrom = Math.min(Math.floor(pos), input.length - 1);
    this.tail = input.slice(keepFrom);
    this.readPos = pos - keepFrom;
    return true;
  }
}

registerProcessor("pcm-processor", PcmProcessor);
