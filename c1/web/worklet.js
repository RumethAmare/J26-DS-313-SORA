// Runs on the browser's audio thread. Turns microphone audio into 16 kHz mono
// float32 blocks of 512 samples (32 ms, the size the server's VAD expects) and
// posts them to the page, which forwards them over the WebSocket.
// If the AudioContext could not be opened at 16 kHz, it downsamples linearly.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;  // input samples per output sample
    this.pos = 0;                     // fractional read position into `carry`
    this.carry = new Float32Array(0);
    this.out = new Float32Array(512);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let src = ch;
    if (this.carry.length) {
      src = new Float32Array(this.carry.length + ch.length);
      src.set(this.carry);
      src.set(ch, this.carry.length);
    }
    let pos = this.pos;
    while (pos + 1 < src.length) {
      const i = Math.floor(pos), f = pos - i;
      this.out[this.n++] = src[i] * (1 - f) + src[i + 1] * f;
      if (this.n === 512) {
        this.port.postMessage(this.out.slice(0));
        this.n = 0;
      }
      pos += this.ratio;
    }
    const keep = Math.floor(pos);
    this.carry = src.slice(keep);
    this.pos = pos - keep;
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
