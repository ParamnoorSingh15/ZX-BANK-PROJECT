const API_BASE = '/api'

export async function askQuestion(question, options = {}) {
  const response = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      strategy: options.strategy || 'hybrid',
      k: options.k || 8,
      use_conversation: options.use_conversation || false,
      session_id: options.session_id || null,
    }),
  })
  if (!response.ok) {
    let detail = `API error: ${response.status}`
    try {
      const body = await response.json()
      if (body.detail) detail = body.detail
    } catch {
      // Keep the status code message when the body is not JSON.
    }
    throw new Error(detail)
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
    let detail = `Upload error: ${response.status}`
    try {
      const body = await response.json()
      if (body.detail) detail = body.detail
    } catch {
      // Keep the status code message.
    }
    throw new Error(detail)
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

export async function clearConversation(sessionId) {
  if (!sessionId) return
  await fetch(`${API_BASE}/conversations/${sessionId}`, { method: 'DELETE' })
}
