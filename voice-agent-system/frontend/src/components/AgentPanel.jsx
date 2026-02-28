import React from 'react';

function AgentCard({ agent, isActive, isTransferTarget, isTransferSource }) {
  return (
    <div
      className={[
        'agent-card',
        isActive ? 'is-active' : '',
        isTransferTarget ? 'is-transfer-target' : '',
        isTransferSource ? 'is-transfer-source' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="agent-avatar">{agent.name.slice(0, 1)}</div>
      <div className="agent-meta">
        <div className="agent-name">{agent.name}</div>
        <div className="agent-title">{agent.title}</div>
      </div>
      <div className="agent-state">{isActive ? 'Speaking' : 'Idle'}</div>
    </div>
  );
}

export default function AgentPanel({ agents, activeAgentId, transfer }) {
  return (
    <section className="agent-panel">
      {agents.map((agent) => (
        <AgentCard
          key={agent.id}
          agent={agent}
          isActive={activeAgentId === agent.id}
          isTransferTarget={transfer?.new_agent === agent.id}
          isTransferSource={transfer?.from_agent === agent.id}
        />
      ))}
    </section>
  );
}
