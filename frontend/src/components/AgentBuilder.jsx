import { useState } from 'react';

const EMPTY_FORM = {
  name: '',
  title: '',
  specialty: '',
  system_prompt: '',
  tts_speaker: 'autumn',
  tts_instruct: '',
  gender: 'male',
};

export default function AgentBuilder({
  agents,
  voices,
  defaultAgentId,
  onClose,
  onCreateAgent,
  onUpdateAgent,
  onDeleteAgent,
  onSetDefault,
}) {
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [isNew, setIsNew] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(null);

  const startNew = () => {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setIsNew(true);
    setError('');
  };

  const startEdit = (agent) => {
    setEditingId(agent.id);
    setForm({
      name: agent.name || '',
      title: agent.title || '',
      specialty: agent.specialty || '',
      system_prompt: agent.system_prompt || '',
      tts_speaker: agent.tts_speaker || 'autumn',
      tts_instruct: agent.tts_instruct || '',
      gender: agent.gender || 'male',
    });
    setIsNew(false);
    setError('');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setIsNew(false);
    setForm(EMPTY_FORM);
    setError('');
  };

  const handleSave = async () => {
    if (!form.name.trim()) {
      setError('Name is required');
      return;
    }
    setSaving(true);
    setError('');
    try {
      if (isNew) {
        await onCreateAgent(form);
      } else {
        await onUpdateAgent(editingId, form);
      }
      cancelEdit();
    } catch (err) {
      setError(err.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (agentId) => {
    try {
      await onDeleteAgent(agentId);
      setConfirmDelete(null);
      if (editingId === agentId) {
        cancelEdit();
      }
    } catch (err) {
      setError(err.message || 'Delete failed');
    }
  };

  const handleSetDefault = async (agentId) => {
    try {
      await onSetDefault(agentId);
    } catch (err) {
      setError(err.message || 'Failed to set default');
    }
  };

  const showForm = isNew || editingId;

  return (
    <div className="agent-builder-overlay">
      <div className="agent-builder">
        <div className="ab-header">
          <h2>Configure Agents</h2>
          <button className="ab-close" onClick={onClose} type="button">&times;</button>
        </div>

        <div className="ab-body">
          {/* Agent list */}
          <div className="ab-agent-list">
            <div className="ab-list-header">
              <h3>Team ({agents.length})</h3>
              <button className="ab-add-btn" onClick={startNew} type="button">+ Add</button>
            </div>
            {agents.map((agent) => (
              <div
                key={agent.id}
                className={`ab-agent-row ${editingId === agent.id ? 'is-editing' : ''}`}
              >
                <div className="ab-agent-info" onClick={() => startEdit(agent)}>
                  <div className="ab-agent-avatar">{agent.name?.slice(0, 1)}</div>
                  <div>
                    <div className="ab-agent-name">
                      {agent.name}
                      {agent.id === defaultAgentId && (
                        <span className="ab-default-badge">Default</span>
                      )}
                    </div>
                    <div className="ab-agent-title">{agent.title}</div>
                    <div className="ab-agent-specialty">{agent.specialty}</div>
                  </div>
                </div>
                <div className="ab-agent-actions">
                  {agent.id !== defaultAgentId && (
                    <button
                      className="ab-btn-sm"
                      onClick={() => handleSetDefault(agent.id)}
                      title="Set as default entry agent"
                      type="button"
                    >
                      Set Default
                    </button>
                  )}
                  <button
                    className="ab-btn-sm ab-btn-edit"
                    onClick={() => startEdit(agent)}
                    type="button"
                  >
                    Edit
                  </button>
                  {confirmDelete === agent.id ? (
                    <>
                      <button
                        className="ab-btn-sm ab-btn-danger"
                        onClick={() => handleDelete(agent.id)}
                        type="button"
                      >
                        Confirm
                      </button>
                      <button
                        className="ab-btn-sm"
                        onClick={() => setConfirmDelete(null)}
                        type="button"
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <button
                      className="ab-btn-sm ab-btn-danger"
                      onClick={() => setConfirmDelete(agent.id)}
                      type="button"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Edit/Create form */}
          {showForm && (
            <div className="ab-form">
              <h3>{isNew ? 'New Agent' : `Edit: ${form.name}`}</h3>

              {error && <div className="ab-form-error">{error}</div>}

              <label className="ab-field">
                <span>Name *</span>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="e.g. Vikram"
                />
              </label>

              <label className="ab-field">
                <span>Title / Role</span>
                <input
                  type="text"
                  value={form.title}
                  onChange={(e) => setForm({ ...form, title: e.target.value })}
                  placeholder="e.g. Backend Developer"
                />
              </label>

              <label className="ab-field">
                <span>Specialty</span>
                <input
                  type="text"
                  value={form.specialty}
                  onChange={(e) => setForm({ ...form, specialty: e.target.value })}
                  placeholder="e.g. APIs, databases, Python, Node.js"
                />
              </label>

              <label className="ab-field">
                <span>Voice</span>
                <select
                  value={form.tts_speaker}
                  onChange={(e) => setForm({ ...form, tts_speaker: e.target.value })}
                >
                  {voices.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.label} ({v.gender})
                    </option>
                  ))}
                </select>
              </label>

              <label className="ab-field">
                <span>Gender</span>
                <div className="ab-radio-group">
                  <label>
                    <input
                      type="radio"
                      name="gender"
                      value="male"
                      checked={form.gender === 'male'}
                      onChange={() => setForm({ ...form, gender: 'male' })}
                    />
                    Male
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="gender"
                      value="female"
                      checked={form.gender === 'female'}
                      onChange={() => setForm({ ...form, gender: 'female' })}
                    />
                    Female
                  </label>
                </div>
              </label>

              <label className="ab-field">
                <span>Voice Instruction</span>
                <input
                  type="text"
                  value={form.tts_instruct}
                  onChange={(e) => setForm({ ...form, tts_instruct: e.target.value })}
                  placeholder="e.g. Speak in a calm, confident tone"
                />
              </label>

              <label className="ab-field">
                <span>System Prompt</span>
                <textarea
                  rows={8}
                  value={form.system_prompt}
                  onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
                  placeholder="Define personality, expertise, and response style. Routing to other agents is handled automatically."
                />
                <small className="ab-hint">
                  Define personality and expertise. Routing between agents is handled automatically.
                </small>
              </label>

              <div className="ab-form-actions">
                <button className="ab-btn-primary" onClick={handleSave} disabled={saving} type="button">
                  {saving ? 'Saving...' : isNew ? 'Create Agent' : 'Save Changes'}
                </button>
                <button className="ab-btn-secondary" onClick={cancelEdit} type="button">
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
