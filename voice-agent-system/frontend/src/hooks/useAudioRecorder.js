import { useCallback, useEffect, useRef, useState } from 'react';

const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4',
];

function pickSupportedMimeType() {
  if (typeof MediaRecorder === 'undefined') {
    return '';
  }

  for (const mime of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported?.(mime)) {
      return mime;
    }
  }

  return '';
}

export default function useAudioRecorder({ onRecorded }) {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState('');
  const [mimeType, setMimeType] = useState('audio/webm');
  const [audioLevel, setAudioLevel] = useState(0);

  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaSourceRef = useRef(null);
  const monitorTimerRef = useRef(null);
  const startedAtRef = useRef(0);
  const lastVoiceAtRef = useRef(0);
  const hasVoiceRef = useRef(false);

  const cleanupMonitor = useCallback(() => {
    if (monitorTimerRef.current) {
      window.clearInterval(monitorTimerRef.current);
      monitorTimerRef.current = null;
    }
  }, []);

  const cleanupAnalyser = useCallback(async () => {
    analyserRef.current = null;
    mediaSourceRef.current = null;
    const ctx = audioContextRef.current;
    audioContextRef.current = null;
    if (ctx && typeof ctx.close === 'function' && ctx.state !== 'closed') {
      try {
        await ctx.close();
      } catch (_) {
        // Ignore close errors from browser/runtime variance.
      }
    }
  }, []);

  const detectSpeechRms = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) {
      return 0;
    }

    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    let sumSquares = 0;
    for (let i = 0; i < data.length; i += 1) {
      const normalized = data[i] / 128 - 1;
      sumSquares += normalized * normalized;
    }
    return Math.sqrt(sumSquares / data.length);
  }, []);

  const ensureStream = useCallback(async () => {
    if (streamRef.current) {
      return streamRef.current;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    streamRef.current = stream;
    return stream;
  }, []);

  const startRecording = useCallback(async () => {
    if (isRecording) {
      return;
    }

    setError('');

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone access is not supported in this browser.');
      return;
    }

    try {
      const stream = await ensureStream();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        // Reuse existing AudioContext to avoid browser limits (~6 contexts)
        let ctx = audioContextRef.current;
        if (!ctx || ctx.state === 'closed') {
          ctx = new AudioCtx();
          audioContextRef.current = ctx;
        } else if (ctx.state === 'suspended') {
          await ctx.resume();
        }
        // Disconnect previous source if any
        if (mediaSourceRef.current) {
          try { mediaSourceRef.current.disconnect(); } catch (_) {}
        }
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.2;
        source.connect(analyser);
        mediaSourceRef.current = source;
        analyserRef.current = analyser;
      }

      const supportedMime = pickSupportedMimeType();
      setMimeType(supportedMime || 'audio/webm');

      const recorder = supportedMime
        ? new MediaRecorder(stream, { mimeType: supportedMime })
        : new MediaRecorder(stream);

      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = (event) => {
        const msg = event?.error?.message || 'MediaRecorder failed.';
        setError(msg);
      };

      recorder.onstop = async () => {
        cleanupMonitor();
        setIsRecording(false);
        const hadVoice = hasVoiceRef.current;

        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || supportedMime || 'audio/webm',
        });
        const durationMs = Math.max(0, performance.now() - startedAtRef.current);

        chunksRef.current = [];
        hasVoiceRef.current = false;
        startedAtRef.current = 0;
        lastVoiceAtRef.current = 0;
        // Disconnect analyser source but keep AudioContext alive for reuse
        if (mediaSourceRef.current) {
          try { mediaSourceRef.current.disconnect(); } catch (_) {}
          mediaSourceRef.current = null;
        }
        analyserRef.current = null;

        if (hadVoice && blob.size > 1200 && durationMs > 250) {
          await onRecorded?.(blob);
        }
      };

      recorderRef.current = recorder;
      recorder.start(250);
      startedAtRef.current = performance.now();
      lastVoiceAtRef.current = startedAtRef.current;
      hasVoiceRef.current = false;

      cleanupMonitor();
      monitorTimerRef.current = window.setInterval(() => {
        const now = performance.now();
        const rms = detectSpeechRms();

        setAudioLevel(Math.min(rms * 10, 1)); // Normalize to 0-1 range

        if (rms > 0.02) {
          hasVoiceRef.current = true;
          lastVoiceAtRef.current = now;
        }

        const recorderNow = recorderRef.current;
        if (!recorderNow || recorderNow.state === 'inactive') {
          return;
        }

        const silenceMs = now - lastVoiceAtRef.current;
        const elapsedMs = now - startedAtRef.current;
        if (hasVoiceRef.current && elapsedMs > 500 && silenceMs > 900) {
          recorderNow.stop();
          return;
        }

        if (!hasVoiceRef.current && elapsedMs > 4000) {
          recorderNow.stop();
          return;
        }

        // Hard cap at 30 seconds to prevent infinite recording from background noise
        if (elapsedMs > 30000) {
          recorderNow.stop();
        }
      }, 100);

      setIsRecording(true);
    } catch (err) {
      setError(err?.message || 'Failed to access microphone.');
    }
  }, [cleanupAnalyser, cleanupMonitor, detectSpeechRms, ensureStream, isRecording, onRecorded]);

  const stopRecording = useCallback(() => {
    cleanupMonitor();
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') {
      return;
    }

    recorder.stop();
  }, [cleanupMonitor]);

  useEffect(() => {
    return () => {
      cleanupMonitor();
      void cleanupAnalyser();
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
      }

      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cleanupAnalyser, cleanupMonitor]);

  return {
    isRecording,
    error,
    mimeType,
    audioLevel,
    startRecording,
    stopRecording,
  };
}
