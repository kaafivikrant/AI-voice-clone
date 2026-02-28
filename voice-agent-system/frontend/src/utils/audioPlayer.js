let audioContext = null;
let queue = Promise.resolve();
let currentSource = null;

function getAudioContext() {
  if (!audioContext) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtx();
  }
  return audioContext;
}

export async function unlockAudio() {
  const ctx = getAudioContext();
  if (ctx.state === 'suspended') {
    await ctx.resume();
  }
}

export function stopPlayback() {
  if (currentSource) {
    try {
      currentSource.stop();
    } catch (_) {}
    currentSource = null;
  }
  // Clear the queue so pending chunks don't play
  queue = Promise.resolve();
}

export function playAudio(wavArrayBuffer) {
  queue = queue
    .then(async () => {
      const ctx = getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      const clone = wavArrayBuffer.slice(0);
      const decoded = await ctx.decodeAudioData(clone);

      await new Promise((resolve) => {
        const source = ctx.createBufferSource();
        source.buffer = decoded;
        source.connect(ctx.destination);
        source.onended = () => {
          currentSource = null;
          resolve();
        };
        currentSource = source;
        source.start();
      });
    })
    .catch((err) => {
      console.error('Audio playback error:', err);
    });

  return queue;
}
