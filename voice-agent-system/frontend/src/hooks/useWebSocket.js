import { useCallback, useEffect, useRef, useState } from 'react';

function resolveWebSocketUrl() {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}/ws/voice`;
}

export default function useWebSocket({ onJsonMessage, onAudioMessage }) {
  const [status, setStatus] = useState('connecting');
  const [reconnectCount, setReconnectCount] = useState(0);

  const wsRef = useRef(null);
  const onJsonMessageRef = useRef(onJsonMessage);
  const onAudioMessageRef = useRef(onAudioMessage);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);

  useEffect(() => {
    onJsonMessageRef.current = onJsonMessage;
  }, [onJsonMessage]);

  useEffect(() => {
    onAudioMessageRef.current = onAudioMessage;
  }, [onAudioMessage]);

  const connect = useCallback(() => {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const ws = new WebSocket(resolveWebSocketUrl());
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    setStatus('connecting');

    ws.onopen = () => {
      setStatus('connected');
      setReconnectCount(0);
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === 'string') {
        try {
          const payload = JSON.parse(event.data);
          onJsonMessageRef.current?.(payload);
        } catch (err) {
          console.error('Failed to parse websocket JSON:', err);
        }
        return;
      }

      if (event.data instanceof ArrayBuffer) {
        onAudioMessageRef.current?.(event.data);
        return;
      }

      if (event.data instanceof Blob) {
        const arrayBuffer = await event.data.arrayBuffer();
        onAudioMessageRef.current?.(arrayBuffer);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setStatus('error');
    };

    ws.onclose = () => {
      wsRef.current = null;
      setStatus('disconnected');

      if (!shouldReconnectRef.current) {
        return;
      }

      setReconnectCount((count) => {
        const next = count + 1;
        const backoffMs = Math.min(5000, 800 + next * 400);

        reconnectTimerRef.current = window.setTimeout(() => {
          connect();
        }, backoffMs);

        return next;
      });
    };
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendJson = useCallback((payload) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    ws.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendAudioBlob = useCallback(
    async (blob) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        return false;
      }

      sendJson({
        type: 'audio_meta',
        mime_type: blob.type || 'audio/webm',
      });

      const audioBuffer = await blob.arrayBuffer();
      ws.send(audioBuffer);
      return true;
    },
    [sendJson]
  );

  const resetSession = useCallback(() => sendJson({ type: 'reset' }), [sendJson]);

  const sendTextInput = useCallback(
    (text) => sendJson({ type: 'text_input', text }),
    [sendJson]
  );

  return {
    status,
    reconnectCount,
    sendJson,
    sendAudioBlob,
    resetSession,
    sendTextInput,
  };
}
