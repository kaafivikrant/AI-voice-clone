import React, { useEffect, useRef } from 'react';

export default function ChatHistory({ messages, processingStage, activeAgentName }) {
  const listRef = useRef(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, processingStage]);

  const isThinking = processingStage === 'thinking' || processingStage === 'transcribing';

  return (
    <section className="chat-history" ref={listRef}>
      {messages.length === 0 && !isThinking && (
        <p className="chat-placeholder">Say something to start the conversation.</p>
      )}

      {messages.map((message) => (
        <article
          key={message.id}
          className={`chat-bubble ${message.role === 'user' ? 'is-user' : 'is-agent'}`}
        >
          <header>
            <span>{message.role === 'user' ? 'You' : message.agentName || 'Agent'}</span>
          </header>
          <p>{message.text}</p>
        </article>
      ))}

      {isThinking && (
        <article className="chat-bubble is-agent is-thinking">
          <header>
            <span>{activeAgentName || 'Agent'}</span>
          </header>
          <p className="thinking-dots">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </p>
        </article>
      )}
    </section>
  );
}
