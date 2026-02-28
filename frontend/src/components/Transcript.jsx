import React from 'react';

export default function Transcript({ text, processingStage }) {
  return (
    <section className="transcript-box">
      <div className="transcript-header">
        <h2>Live Transcript</h2>
        <span>{processingStage || 'idle'}</span>
      </div>
      <p>{text || 'Your transcribed speech will appear here.'}</p>
    </section>
  );
}
