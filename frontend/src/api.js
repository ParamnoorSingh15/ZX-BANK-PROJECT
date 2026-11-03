const API_BASE = '/api'

export async function askQuestion(question, options = {}) {
  const response = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      strategy: options.strategy || 'dense',
      k: options.k || 4,
      use_conversation: options.use_conversation || false,
      session_id: options.session_id || null,
    }),
  })
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }
  return response.json()
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`Upload error: ${response.status}`)
  }
  return response.json()
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new Error(`Health error: ${response.status}`)
  }
  return response.json()
}
