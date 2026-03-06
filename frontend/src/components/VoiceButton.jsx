export default function VoiceButton({ isRecording, isContinuous, disabled, onToggle, connectionStatus, audioLevel }) {
  const handleClick = async () => {
    if (!disabled) {
      await onToggle?.();
    }
  };

  const statusLabel = disabled
    ? connectionStatus === 'connected'
      ? 'Unavailable'
      : 'Connecting'
    : isRecording
      ? 'Listening...'
      : 'Speak';

  const ringScale = isRecording && audioLevel ? 1 + Math.min(audioLevel, 1) * 0.3 : 1;

  return (
    <div className="voice-button-wrap">
      <button
        className={`voice-button ${isRecording ? 'is-recording' : ''}`}
        type="button"
        disabled={disabled}
        onClick={handleClick}
      >
        {isRecording ? 'Stop' : 'Mic'}
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
