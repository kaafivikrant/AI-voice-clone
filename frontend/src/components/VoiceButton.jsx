import React from 'react';

export default function VoiceButton({ isRecording, isContinuous, disabled, onToggle, connectionStatus, audioLevel }) {
  const handleClick = async () => {
    if (!disabled) {
      await onToggle?.();
    }
  };

  const statusLabel = disabled
    ? connectionStatus === 'connected'
      ? 'Mic unavailable'
      : 'Connecting...'
    : isRecording
      ? 'Listening... stop speaking to send'
      : isContinuous
        ? 'Auto-listen active (waiting for next turn)'
        : 'Click once to start continuous listening';

  // Scale the ring from 1.0 to 1.4 based on audio level
  const ringScale = isRecording && audioLevel ? 1 + Math.min(audioLevel, 1) * 0.4 : 1;

  return (
    <div className="voice-button-wrap">
      <button
        className={`voice-button ${isRecording ? 'is-recording' : ''}`}
        type="button"
        disabled={disabled}
        onClick={handleClick}
      >
        {isContinuous ? 'Stop' : 'Start'}
        <span
          className="voice-ring"
          aria-hidden="true"
          style={isRecording ? { transform: `scale(${ringScale})` } : undefined}
        />
      </button>
      <p className="voice-caption">{statusLabel}</p>
    </div>
  );
}
