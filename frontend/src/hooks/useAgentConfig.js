import { useCallback, useEffect, useState } from 'react';

function adminHeaders(extra = {}) {
  const key = import.meta.env.VITE_ADMIN_API_KEY || '';
  const headers = { ...extra };
  if (key) {
    headers['X-API-Key'] = key;
  }
  return headers;
}

export default function useAgentConfig() {
  const [agents, setAgents] = useState([]);
  const [voices, setVoices] = useState([]);
  const [defaultAgentId, setDefaultAgentId] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchAgents = useCallback(async () => {
    try {
      const key = import.meta.env.VITE_ADMIN_API_KEY || '';
      const url = key ? '/api/agents?full=true' : '/api/agents';
      const res = await fetch(url, { headers: adminHeaders() });
      const data = await res.json();
      setAgents(data.agents || []);
      setDefaultAgentId(data.default_agent_id || '');
      setError('');
    } catch (err) {
      setError('Failed to load agents');
      console.error('fetchAgents error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchVoices = useCallback(async () => {
    try {
      const res = await fetch('/api/voices');
      const data = await res.json();
      setVoices(data.voices || []);
    } catch (err) {
      console.error('fetchVoices error:', err);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
    fetchVoices();
  }, [fetchAgents, fetchVoices]);

  const createAgent = useCallback(async (agentData) => {
    const res = await fetch('/api/agents', {
      method: 'POST',
      headers: adminHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(agentData),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to create agent');
    }
    await fetchAgents();
    return res.json();
  }, [fetchAgents]);

  const updateAgent = useCallback(async (agentId, agentData) => {
    const res = await fetch(`/api/agents/${agentId}`, {
      method: 'PUT',
      headers: adminHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(agentData),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to update agent');
    }
    await fetchAgents();
  }, [fetchAgents]);

  const deleteAgent = useCallback(async (agentId) => {
    const res = await fetch(`/api/agents/${agentId}`, {
      method: 'DELETE',
      headers: adminHeaders(),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to delete agent');
    }
    await fetchAgents();
  }, [fetchAgents]);

  const generatePersonality = useCallback(async (agentId, prompt = '') => {
    const res = await fetch(`/api/agents/${agentId}/generate-personality`, {
      method: 'POST',
      headers: adminHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to generate personality');
    }
    const data = await res.json();
    return data.personality_json;
  }, []);

  const setDefault = useCallback(async (agentId) => {
    const res = await fetch(`/api/agents/default/${agentId}`, {
      method: 'PUT',
      headers: adminHeaders(),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to set default');
    }
    await fetchAgents();
  }, [fetchAgents]);

  return {
    agents,
    voices,
    defaultAgentId,
    loading,
    error,
    createAgent,
    updateAgent,
    deleteAgent,
    setDefault,
    generatePersonality,
    refreshAgents: fetchAgents,
  };
}
