class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._bufferSize = 4800; // send every 300ms at 16kHz
    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "set_config") {
        const chunkMs = Number(msg.audioChunkMs);
        if (Number.isFinite(chunkMs)) {
          const clamped = Math.min(1000, Math.max(100, chunkMs));
          this._bufferSize = Math.max(1, Math.round((16000 * clamped) / 1000));
        }
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];
    const inputSampleRate = sampleRate;
    const outputSampleRate = 16000;
    const ratio = inputSampleRate / outputSampleRate;

    for (let i = 0; i < channelData.length; i += ratio) {
      this._buffer.push(channelData[Math.floor(i)]);
    }

    if (this._buffer.length >= this._bufferSize) {
      const chunk = new Float32Array(this._buffer);
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
      this._buffer = [];
    }

    return true;
  }
}

registerProcessor("audio-stream-processor", AudioStreamProcessor);
