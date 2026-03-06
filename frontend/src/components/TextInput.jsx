import React, { useState } from 'react';

export default function TextInput({ disabled, onSend }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
  };

  return (
    <form className="text-input-form" onSubmit={handleSubmit}>
      <input
        className="text-input-field"
        type="text"
        placeholder="Ask your team something..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
      />
      <button
        className="text-input-send"
        type="submit"
        disabled={disabled || !text.trim()}
      >
        Send
      </button>
    </form>
  );
}
