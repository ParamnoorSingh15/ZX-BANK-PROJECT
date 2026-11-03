import { useState, useEffect, useRef } from 'react'
import { askQuestion, uploadDocument, getHealth } from './api.js'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [strategy, setStrategy] = useState('dense')
  const [sessionId, setSessionId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [useConversation, setUseConversation] = useState(false)
  const [newConversation, setNewConversation] = useState(false)
  const [conversations, setConversations] = useState([])
  const [currentConvId, setCurrentConvId] = useState(null)
  const inputRef = useRef(null)
  const messagesEndRef = useRef(null)

  const exampleQuestions = [
    'Where is ZX Bank headquartered?',
    'What are the savings account features?',
    'What is Zia in the mobile app?',
    'Does ZX Bank offer home loans?',
  ]

  const handleExampleClick = (q) => {
    setInput(q)
    inputRef.current?.focus()
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    let mounted = true
    getHealth()
      .then((h) => {
        if (mounted) setStatus(h)
      })
      .catch((err) => {
        if (mounted) setStatus({ status: 'error', error: err.message })
      })
    return () => { mounted = false }
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    const userMsg = { role: 'user', content: question, sources: [] }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setError(null)
    setLoading(true)

    try {
      const data = await askQuestion(question, {
        strategy,
        k: 4,
        use_conversation: useConversation,
        session_id: sessionId || undefined,
      })
      if (!sessionId && useConversation && data.session_id) {
        setSessionId(data.session_id)
      }
      const botMsg = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
      }
      setMessages((prev) => [...prev, botMsg])
    } catch (err) {
      setError(err.message)
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Error: ' + err.message, sources: [] }])
    } finally {
      setLoading(false)
    }
  }

  const handleNewConversation = () => {
    setNewConversation(true)
    setSessionId(null)
    setMessages([])
    setError(null)
  }

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const result = await uploadDocument(file)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Uploaded "${result.filename}" and indexed ${result.chunks_indexed} new chunks into collection "${result.collection}".`,
          sources: [],
        },
      ])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      e.target.value = ''
    }
  }

  const formatMeta = (s) => {
    const parts = []
    if (s.format) parts.push(s.format)
    if (s.page) parts.push(`p. ${s.page}`)
    if (s.slide) parts.push(`slide ${s.slide}`)
    return parts.length ? ' (' + parts.join(', ') + ')' : ''
  }

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>ZX Bank</h1>
          <p>Zia &mdash; AI Banking Assistant</p>
        </div>
        <div className="new-chat" onClick={handleNewConversation}>
          + New conversation
        </div>
        <div className="conversations">
          {conversations.length === 0 && (
            <div style={{ padding: '1rem', color: '#9ca3af', fontSize: '0.9rem' }}>
              No conversations yet
            </div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`conversation-item ${currentConvId === c.id ? 'active' : ''}`}
              onClick={() => setCurrentConvId(c.id)}
            >
              {c.title}
            </div>
          ))}
        </div>
      </div>

      <div className="main">
        <div className="header">
          <h2>Zia Assistant</h2>
        </div>

        <div className="chat-area">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">🏦</div>
              <h3>Ask me about ZX Bank</h3>
              <p>Grounded answers from ZX Bank documents with source citations.</p>
              <div className="example-chips">
                {exampleQuestions.map((q, i) => (
                  <div key={i} className="example-chip" onClick={() => handleExampleClick(q)}>
                    {q}
                  </div>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="avatar">
                {msg.role === 'user' ? 'You' : 'Zia'}
              </div>
              <div className="message-content">
                {msg.content}
                {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <div className="sources-title">Sources</div>
                    {msg.sources.map((s, si) => (
                      <div key={si} className="source-item">
                        <span className="source-title">{s.title}</span>
                        <span className="source-meta">{formatMeta(s)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {error && <div className="error-msg">{error}</div>}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <div className="input-wrapper">
            <select
              className="strategy-select"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              <option value="dense">Dense search</option>
              <option value="mmr">MMR (diverse)</option>
              <option value="filtered">Filtered</option>
            </select>
            <label className="conversation-toggle">
              <input
                type="checkbox"
                checked={useConversation}
                onChange={(e) => setUseConversation(e.target.checked)}
              />
              Remember conversation
            </label>
            <input
              ref={inputRef}
              className="text-input"
              placeholder="Ask a question about ZX Bank..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit(e)}
              disabled={loading}
            />
            <button className="send-btn" onClick={handleSubmit} disabled={loading}>
              {loading ? 'Asking…' : 'Ask'}
            </button>
          </div>
          <div style={{ marginTop: '0.75rem', maxWidth: 1000, margin: '0 auto 0 auto', display: 'flex', justifyContent: 'center' }}>
            <label className="upload-btn">
              📎 Upload a document
              <input
                type="file"
                accept=".pdf,.html,.docx,.pptx,.md"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </label>
          </div>

          <div className={`status-bar ${status?.status === 'ok' ? 'ok' : status?.status === 'error' ? 'err' : 'connecting'}`}>
            {status ? (
              <>
                <span className="status-dot" aria-hidden="true">●</span>
                {status.status === 'ok' ? (
                  <>
                    Qdrant connected &nbsp;•&nbsp; {status.collection_exists ? 'collection ready' : 'no collection'} &nbsp;•&nbsp; {status.vector_count || 0} vectors
                  </>
                ) : (
                  <span>Backend unreachable — start the FastAPI server</span>
                )}
              </>
            ) : (
              <span>Connecting…</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
