const STAGE_LABELS = {
  idle: 'Ready',
  uploading: 'Sending',
  transcribing: 'Listening',
  thinking: 'Reasoning',
  speaking: 'Speaking',
  processing: 'Processing',
};

export default function Transcript({ text, processingStage }) {
  const label = STAGE_LABELS[processingStage] || STAGE_LABELS.idle;

  return (
    <section className="transcript-box">
      <div className="transcript-header">
        <h2>Transcript</h2>
        <span>{label}</span>
      </div>
      <p>{text || 'Your speech will appear here as you talk.'}</p>
    </section>
  );
}
