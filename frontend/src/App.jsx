import { useCallback, useMemo, useRef, useState } from 'react';
import AgentBuilder from './components/AgentBuilder';
import AgentPanel from './components/AgentPanel';
import ChatHistory from './components/ChatHistory';
import TextInput from './components/TextInput';
import Transcript from './components/Transcript';
import VoiceButton from './components/VoiceButton';
import useAgentConfig from './hooks/useAgentConfig';
import useAudioRecorder from './hooks/useAudioRecorder';
import useWebSocket from './hooks/useWebSocket';
import { playAudio, stopPlayback, unlockAudio } from './utils/audioPlayer';

function createMessage({ id, role, agentId, agentName, text }) {
  return {
    id: id || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    agentId,
    agentName,
    text,
  };
}

function isThanksOnly(text) {
  const normalized = (text || '')
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return normalized === 'thank you' || normalized === 'thanks' || normalized === 'thank you so much';
}

export default function App() {
  const agentConfig = useAgentConfig();
  const [agents, setAgents] = useState([]);
  const [activeAgentId, setActiveAgentId] = useState('');
  const [defaultAgentId, setDefaultAgentId] = useState('');
  const [messages, setMessages] = useState([]);
  const [transcript, setTranscript] = useState('');
  const [processingStage, setProcessingStage] = useState('idle');
  const [transfer, setTransfer] = useState(null);
  const [serverError, setServerError] = useState('');
  const [continuousMode, setContinuousMode] = useState(false);
  const [showBuilder, setShowBuilder] = useState(false);
  const continuousModeRef = useRef(false);

  // Track the streaming message being built
  const streamingMsgIdRef = useRef(null);
  const pendingAudioCountRef = useRef(0);

  const agentNameById = useMemo(() => {
    const map = new Map();
    agents.forEach((agent) => map.set(agent.id, agent.name));
    return map;
  }, [agents]);

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  const updateStreamingMessage = useCallback((msgId, agentId, agentName, newText) => {
    setMessages((prev) => {
      const existing = prev.find((m) => m.id === msgId);
      if (existing) {
        return prev.map((m) =>
          m.id === msgId ? { ...m, text: m.text + newText } : m
        );
      }
      return [
        ...prev,
        createMessage({ id: msgId, role: 'assistant', agentId, agentName, text: newText }),
      ];
    });
  }, []);

  const ws = useWebSocket({
    onJsonMessage: (payload) => {
      if (!payload?.type) {
        return;
      }

      if (payload.type === 'ready') {
        if (Array.isArray(payload.agents) && payload.agents.length > 0) {
          setAgents(payload.agents);
        }
        if (payload.current_agent) {
          setActiveAgentId(payload.current_agent);
        }
        if (payload.default_agent_id) {
          setDefaultAgentId(payload.default_agent_id);
        }
        setServerError('');
        return;
      }

      if (payload.type === 'agents_updated') {
        if (Array.isArray(payload.agents) && payload.agents.length > 0) {
          setAgents(payload.agents);
        }
        if (payload.default_agent_id) {
          setDefaultAgentId(payload.default_agent_id);
        }
        agentConfig.refreshAgents();
        return;
      }

      if (payload.type === 'transcript') {
        const userText = payload.text || '';
        setTranscript(userText);
        appendMessage(
          createMessage({
            role: 'user',
            text: userText,
          })
        );
        if (isThanksOnly(userText)) {
          setContinuousMode(false);
          continuousModeRef.current = false;
        }
        return;
      }

      if (payload.type === 'processing') {
        setProcessingStage(payload.stage || 'processing');
        return;
      }

      // Streaming: partial sentence chunk
      if (payload.type === 'response_chunk') {
        const agentId = payload.agent || activeAgentId;
        if (!streamingMsgIdRef.current) {
          streamingMsgIdRef.current = `stream-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        }
        pendingAudioCountRef.current += 1;
        updateStreamingMessage(
          streamingMsgIdRef.current,
          agentId,
          agentNameById.get(agentId),
          (streamingMsgIdRef.current ? ' ' : '') + payload.text,
        );
        setProcessingStage('speaking');
        return;
      }

      // Streaming: response complete
      if (payload.type === 'response_end') {
        streamingMsgIdRef.current = null;
        setProcessingStage('idle');
        return;
      }

      // Legacy non-streaming response
      if (payload.type === 'response') {
        const agentId = payload.agent || activeAgentId;
        appendMessage(
          createMessage({
            role: 'assistant',
            agentId,
            agentName: agentNameById.get(agentId),
            text: payload.text || '',
          })
        );
        setProcessingStage('idle');
        return;
      }

      if (payload.type === 'agent_state') {
        if (payload.current_agent) {
          setActiveAgentId(payload.current_agent);
        }
        return;
      }

      if (payload.type === 'escalation') {
        setTransfer({
          from_agent: payload.from_agent,
          new_agent: payload.new_agent,
        });
        if (payload.new_agent) {
          setActiveAgentId(payload.new_agent);
        }

        window.setTimeout(() => {
          setTransfer(null);
        }, 1600);
        return;
      }

      if (payload.type === 'error') {
        setServerError(payload.message || 'Unexpected server error.');
        setProcessingStage('idle');
        const isNoSpeechError = (payload.message || '').toLowerCase().includes("couldn't hear");
        if (isNoSpeechError) {
          setContinuousMode(false);
          continuousModeRef.current = false;
          return;
        }
        if (continuousModeRef.current) {
          window.setTimeout(() => {
            recorder.startRecording();
          }, 300);
        }
      }
    },
    onAudioMessage: async (audioBuffer) => {
      try {
        await playAudio(audioBuffer);
        pendingAudioCountRef.current = Math.max(0, pendingAudioCountRef.current - 1);
        if (continuousModeRef.current && pendingAudioCountRef.current === 0) {
          await recorder.startRecording();
        }
      } catch (err) {
        setServerError(err?.message || 'Failed to play audio response.');
      }
    },
  });

  const recorder = useAudioRecorder({
    onRecorded: async (blob) => {
      setProcessingStage('uploading');
      setServerError('');

      const ok = await ws.sendAudioBlob(blob);
      if (!ok) {
        setServerError('WebSocket is not connected.');
        setProcessingStage('idle');
      }
    },
  });

  const handleStartRecording = async () => {
    setServerError('');
    stopPlayback();
    pendingAudioCountRef.current = 0;
    await unlockAudio();
    await recorder.startRecording();
  };

  const handleToggleRecording = async () => {
    if (recorder.isRecording) {
      setContinuousMode(false);
      continuousModeRef.current = false;
      recorder.stopRecording();
      return;
    }
    setContinuousMode(true);
    continuousModeRef.current = true;
    await handleStartRecording();
  };

  const handleTextInput = (text) => {
    setServerError('');
    ws.sendTextInput(text);
  };

  const handleReset = async () => {
    ws.resetSession();
    setActiveAgentId(defaultAgentId || '');
    setMessages([]);
    setTranscript('');
    setTransfer(null);
    setProcessingStage('idle');
    setServerError('');
    setContinuousMode(false);
    continuousModeRef.current = false;
    streamingMsgIdRef.current = null;
    pendingAudioCountRef.current = 0;

    try {
      await fetch('/api/reset', { method: 'POST' });
    } catch (err) {
      console.warn('HTTP reset failed:', err);
    }
  };

  const defaultAgentName = agentNameById.get(defaultAgentId) || 'Default';

  const statusText =
    ws.status === 'connected'
      ? 'Connected'
      : ws.status === 'connecting'
        ? 'Connecting...'
        : ws.status === 'error'
          ? 'Connection error'
          : `Disconnected (${ws.reconnectCount})`;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>AI Agent Team</h1>
          <p>Intelligent routing between specialists</p>
        </div>
        <div className="topbar-actions">
          <button className="config-btn" type="button" onClick={() => setShowBuilder(true)}>
            Configure Agents
          </button>
          <button className="reset-btn" type="button" onClick={handleReset}>
            New Conversation
          </button>
        </div>
      </header>

      <AgentPanel agents={agents} activeAgentId={activeAgentId} transfer={transfer} />

      <main className="content-grid">
        <Transcript text={transcript} processingStage={processingStage} />
        <ChatHistory
          messages={messages}
          processingStage={processingStage}
          activeAgentName={agentNameById.get(activeAgentId)}
        />
      </main>

      <footer className="controls">
        <VoiceButton
          isRecording={recorder.isRecording}
          isContinuous={continuousMode}
          disabled={ws.status !== 'connected'}
          connectionStatus={ws.status}
          onToggle={handleToggleRecording}
          audioLevel={recorder.audioLevel}
        />

        <TextInput
          disabled={ws.status !== 'connected' || processingStage !== 'idle'}
          onSend={handleTextInput}
        />

        <div className="status-block">
          <p>
            Status: <strong>{statusText}</strong>
          </p>
          <p>
            Agent: <strong>{agentNameById.get(activeAgentId) || activeAgentId}</strong>
          </p>
          <p>
            Listening: <strong>{continuousMode ? 'Active' : 'Off'}</strong>
          </p>
        </div>
      </footer>

      {(recorder.error || serverError) && (
        <aside className="error-banner">{recorder.error || serverError}</aside>
      )}

      {showBuilder && (
        <AgentBuilder
          agents={agentConfig.agents}
          voices={agentConfig.voices}
          defaultAgentId={agentConfig.defaultAgentId}
          onClose={() => setShowBuilder(false)}
          onCreateAgent={agentConfig.createAgent}
          onUpdateAgent={agentConfig.updateAgent}
          onDeleteAgent={agentConfig.deleteAgent}
          onSetDefault={agentConfig.setDefault}
        />
      )}
    </div>
  );
}
