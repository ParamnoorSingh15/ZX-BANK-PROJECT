import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  askQuestion,
  uploadDocument,
  getHealth,
  clearConversation,
} from './api.js'

const EXAMPLE_QUESTIONS = [
  'What are the interest rates on fixed deposits?',
  'What is the minimum balance for a savings account?',
  'How do I apply for a personal loan?',
  'How can I block my lost credit card?',
]

const STRATEGY_LABELS = {
  hybrid: 'Hybrid (recommended)',
  dense: 'Dense vectors',
  mmr: 'MMR (diverse)',
}

function formatSourceMeta(source) {
  const parts = []
  if (source.section) parts.push(source.section)
  if (source.page) parts.push(`p. ${source.page}`)
  if (source.slide) parts.push(`slide ${source.slide}`)
  if (source.format) parts.push(source.format.toUpperCase())
  return parts.length ? `(${parts.join(' · ')})` : ''
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [askLoading, setAskLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [strategy, setStrategy] = useState('hybrid')
  const [sessionId, setSessionId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [useConversation, setUseConversation] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const inputRef = useRef(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, askLoading, uploading])

  useEffect(() => {
    if (!askLoading) {
      setElapsed(0)
      return undefined
    }
    const timer = setInterval(() => setElapsed((value) => value + 1), 1000)
    return () => clearInterval(timer)
  }, [askLoading])

  useEffect(() => {
    let mounted = true

    const checkHealth = () => {
      getHealth()
        .then((health) => {
          if (mounted) setStatus(health)
        })
        .catch((err) => {
          if (mounted) setStatus({ status: 'error', error: err.message })
        })
    }

    checkHealth()
    // Keep polling so the status pill recovers when the backend starts.
    const interval = setInterval(checkHealth, 10000)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [])

  const isOnline = status?.status === 'ok'

  const handleSubmit = async (e) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || askLoading || uploading) return

    const userMsg = { role: 'user', content: question, sources: [] }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setError(null)
    setAskLoading(true)

    try {
      const data = await askQuestion(question, {
        strategy,
        k: 8,
        use_conversation: useConversation,
        session_id: sessionId || undefined,
      })
      if (data.session_id) setSessionId(data.session_id)
      const botMsg = {
        role: 'assistant',
        content: data.answer || 'I could not generate an answer. Please try again.',
        sources: data.sources || [],
      }
      setMessages((prev) => [...prev, botMsg])
    } catch (err) {
      setError(err.message)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            'I could not reach the ZX Bank knowledge base. Make sure Qdrant, ' +
            'the FastAPI backend and Ollama are running, then try again.',
          sources: [],
        },
      ])
    } finally {
      setAskLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleNewConversation = async () => {
    if (sessionId) {
      try {
        await clearConversation(sessionId)
      } catch {
        // Clearing server history is best-effort.
      }
    }
    setSessionId(null)
    setMessages([])
    setError(null)
    setInput('')
    inputRef.current?.focus()
  }

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const result = await uploadDocument(file, sessionId || undefined)
      if (result.session_id) setSessionId(result.session_id)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            `Uploaded **${result.filename}** and indexed ` +
            `**${result.chunks_indexed}** new chunks into ` +
            `\`${result.collection}\`.`,
          sources: [],
        },
      ])
      const health = await getHealth()
      setStatus(health)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  const handleExampleClick = (question) => {
    setInput(question)
    inputRef.current?.focus()
  }

  const statusClass = isOnline ? 'ok' : status ? 'err' : 'connecting'
  const statusLabel = isOnline
    ? 'Knowledge base ready'
    : status
      ? 'Backend offline'
      : 'Connecting'

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-mark">Z</div>
            <div>
              <h1>ZX Bank</h1>
              <p>Zia · AI Banking Assistant</p>
            </div>
          </div>
        </div>

        <button className="new-chat" onClick={handleNewConversation} type="button">
          <span className="plus">+</span>
          New conversation
        </button>

        <div className="conversations">
          <div className="sidebar-section-title">Knowledge base</div>
          <div className="kb-card">
            <div className="kb-row">
              <span>Status</span>
              <span className={`kb-value ${statusClass}`}>{statusLabel}</span>
            </div>
            <div className="kb-row">
              <span>Chunks indexed</span>
              <span className="kb-value">
                {isOnline ? (status.vector_count || 0).toLocaleString() : '—'}
              </span>
            </div>
            <div className="kb-row">
              <span>Collection</span>
              <span className="kb-value mono">
                {isOnline ? status.collection : '—'}
              </span>
            </div>
          </div>

          <div className="sidebar-section-title examples-title">Try asking</div>
          <div className="sidebar-examples">
            {EXAMPLE_QUESTIONS.map((question) => (
              <button
                key={question}
                type="button"
                className="sidebar-example"
                onClick={() => handleExampleClick(question)}
              >
                {question}
              </button>
            ))}
          </div>

          <div className="sidebar-note">
            Answers are grounded in ZX Bank documents and cite the source
            document, section and page/slide.
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="header">
          <div className="header-icon">Z</div>
          <div className="header-titles">
            <h2>Zia Assistant</h2>
            <p>Detailed answers from the ZX Bank knowledge base</p>
          </div>
          <div className={`health-pill ${statusClass}`}>
            <span className="health-dot" aria-hidden="true" />
            {statusLabel}
          </div>
        </header>

        <section className="chat-area" aria-live="polite">
          {messages.length === 0 && !askLoading && !uploading && (
            <div className="empty-state">
              <div className="empty-icon">🏦</div>
              <span className="empty-kicker">ZX Bank · Retrieval-Augmented Generation</span>
              <h3>Ask me anything about ZX Bank</h3>
              <p>
                I search every PDF, HTML, DOCX, PPTX and Markdown document,
                then answer with detailed, cited information.
              </p>
              <div className="example-chips">
                {EXAMPLE_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    className="example-chip"
                    onClick={() => handleExampleClick(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="avatar">{msg.role === 'user' ? 'You' : 'Zia'}</div>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  <div className="markdown-body">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ node, ...props }) => (
                          <a {...props} target="_blank" rel="noreferrer" />
                        ),
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="user-text">{msg.content}</p>
                )}

                {msg.role === 'assistant' &&
                  msg.sources &&
                  msg.sources.length > 0 && (
                    <div className="sources">
                      <div className="sources-title">
                        Sources
                        <span className="source-count">{msg.sources.length}</span>
                      </div>
                      {msg.sources.map((source, sourceIndex) => (
                        <div
                          key={sourceIndex}
                          className="source-item"
                          title={source.content_preview}
                        >
                          <span className="source-title">{source.title}</span>
                          <span className="source-meta">
                            {formatSourceMeta(source)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
              </div>
            </div>
          ))}

          {askLoading && (
            <div className="message bot">
              <div className="avatar">Zia</div>
              <div className="message-content thinking-bubble">
                <div className="typing" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
                <div className="thinking-copy">
                  <strong>Searching ZX Bank documents…</strong>
                  <span>
                    Qwen3 is reasoning over the retrieved context
                    {elapsed > 0 ? ` · ${elapsed}s` : ''}
                  </span>
                </div>
              </div>
            </div>
          )}
          {uploading && (
            <div className="message bot">
              <div className="avatar">Zia</div>
              <div className="message-content thinking-bubble">
                <div className="typing" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
                <div className="thinking-copy">
                  <strong>Indexing uploaded document…</strong>
                  <span>
                    Extracting text, tables and embedding chunks into Qdrant
                  </span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-msg">
              <strong>Connection problem.</strong> {error}
              <button
                type="button"
                className="retry-btn"
                onClick={() => {
                  setError(null)
                  getHealth()
                    .then(setStatus)
                    .catch((err) => setStatus({ status: 'error', error: err.message }))
                }}
              >
                Retry
              </button>
            </div>
          )}
          <div ref={messagesEndRef} />
        </section>

        <div className="input-area">
          <form className="input-wrapper" onSubmit={handleSubmit}>
            <select
              className="strategy-select"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              title="Retrieval strategy"
            >
              {Object.entries(STRATEGY_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
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
              placeholder="Ask a question about ZX Bank…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e)
              }}
              disabled={askLoading || uploading}
            />

            <button className="send-btn" type="submit" disabled={askLoading || uploading}>
              {askLoading ? 'Thinking…' : uploading ? 'Indexing…' : 'Ask'}
            </button>
          </form>

          <div className="input-tools">
            <label className="upload-btn">
              📎 Upload a document
              <input
                type="file"
                accept=".pdf,.html,.htm,.docx,.pptx,.md"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </label>
            <span className="input-hint">
              Detailed answers · citations included · powered by Qwen3 + Qdrant
            </span>
          </div>

          <div className={`status-bar ${statusClass}`}>
            <span className="status-dot" aria-hidden="true">●</span>
            {isOnline ? (
              <span>
                Qdrant connected &nbsp;•&nbsp;{' '}
                {status.collection_exists ? 'collection ready' : 'no collection'}{' '}
                &nbsp;•&nbsp; {(status.vector_count || 0).toLocaleString()} chunks
              </span>
            ) : status ? (
              <span>Backend unreachable — start Qdrant and the FastAPI server</span>
            ) : (
              <span>Connecting to the backend…</span>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
