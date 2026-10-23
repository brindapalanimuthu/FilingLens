import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

const SUGGESTIONS = [
  "What was Apple's total net sales in 2024?",
  "What was Microsoft's net income in fiscal 2025?",
  "How did Apple's stock perform against the S&P 500 over the five years to 2025?",
  "Why did Apple's Services gross margin change?",
  "What was Alphabet's revenue in 2024?",
]

// turn "[Source 2]" / "[2]" / "[Sources 1, 3]" into markdown links to #cite-N,
// which the markdown renderer below shows as clickable superscripts
function linkCitations(text) {
  return (text || '').replace(/\[(?:Sources?\s*)?(\d+(?:\s*,\s*\d+)*)\]/g, (_, ids) =>
    ids
      .split(',')
      .map(s => s.trim())
      .map(id => `[\\[${id}\\]](#cite-${id})`)
      .join(' ')
  )
}

// renders {columns: [...], rows: [{label, values: [...]}]} as a real table
function SourceTable({ table }) {
  const cell = { padding: '8px 12px', borderBottom: '1px solid #e5e7eb', textAlign: 'right', whiteSpace: 'nowrap' }
  const head = { ...cell, fontWeight: 600, background: '#f3f4f6' }
  return (
    <div style={{ overflowX: 'auto', margin: '12px 0' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 14, fontFamily: 'inherit' }}>
        <thead>
          <tr>
            <th style={{ ...head, textAlign: 'left' }}>Line item</th>
            {table.columns.map((c, i) => <th key={i} style={head}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r, i) => (
            <tr key={i}>
              <td style={{ ...cell, textAlign: 'left', whiteSpace: 'normal' }}>{r.label}</td>
              {r.values.map((v, j) => <td key={j} style={cell}>{v}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Chat() {
  const [q, setQ] = useState('')
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [openId, setOpenId] = useState(null)

  async function ask(question) {
    if (!question.trim() || loading) return
    setQ(question)
    setLoading(true)
    setError('')
    setOpenId(null)
    try {
      const r = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setRes(await r.json())
    } catch (e) {
      setError(String(e.message || e))
      setRes(null)
    }
    setLoading(false)
  }

  function openSource(id) {
    setOpenId(id)
    setTimeout(() => document.getElementById(`src-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
  }

  const markdownComponents = {
    a: ({ href, children }) => {
      if (href && href.startsWith('#cite-')) {
        const id = Number(href.replace('#cite-', ''))
        return (
          <sup>
            <button type="button" className="cite" onClick={() => openSource(id)}>[{id}]</button>
          </sup>
        )
      }
      return <a href={href} target="_blank" rel="noreferrer">{children}</a>
    },
  }

  const ver = res?.verification || []

  return (
    <main className="ask-wrap">
      <h1 className="ask-title">FilingLens</h1>
      <p className="ask-sub">
        Ask about Apple and Microsoft 10-K and 10-Q filings. Every number in the answer is
        checked against the source text it came from.
      </p>

      <form className="ask-form" onSubmit={e => { e.preventDefault(); ask(q) }}>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Ask a question..." />
        <button className="ask-btn" disabled={loading}>{loading ? 'Thinking…' : 'Ask'}</button>
      </form>

      <div className="ask-chips">
        {SUGGESTIONS.map(s => <button key={s} onClick={() => ask(s)}>{s}</button>)}
      </div>

      {error && <div className="ask-error">{error}</div>}

      {res && (
        <section className="result">
          <div className="result-card">
            <ReactMarkdown components={markdownComponents}>
              {linkCitations(res.answer)}
            </ReactMarkdown>
          </div>
          <div className="meta">Route: {res.route?.replace('_', ' ')} · Model: {res.model}</div>

          <h3>Number check</h3>
          <div className="checks">
            {ver.map((v, i) => (
              <span key={i} className={v.verified ? 'chip ok' : 'chip bad'}>
                {v.number} {v.verified ? 'found in sources' : 'not found in sources'}
              </span>
            ))}
          </div>
          <p className="meta">
            {res.unverified_count === 0
              ? 'Every number in the answer appears in the retrieved sources.'
              : `${res.unverified_count} number(s) could not be found in the retrieved sources.`}
          </p>

          <h3>Sources</h3>
          {res.sources?.map(s => (
            <details key={s.id} id={`src-${s.id}`} open={openId === s.id}
              onToggle={e => { if (!e.target.open && openId === s.id) setOpenId(null) }}>
              <summary onClick={e => { e.preventDefault(); setOpenId(openId === s.id ? null : s.id) }}>
                <b>[{s.id}]</b> {s.company} FY{s.fiscal_year} · {s.section} <span className="tag">{s.type}</span>
              </summary>
              {s.image_url && (
                <a href={s.image_url} target="_blank" rel="noreferrer" title="Open full size">
                  <img
                    src={s.image_url}
                    alt={`${s.company} FY${s.fiscal_year} ${s.section}`}
                    style={{
                      display: 'block',
                      width: '100%',
                      maxWidth: 720,
                      margin: '12px 0',
                      border: '1px solid #e5e7eb',
                      borderRadius: 8,
                      background: '#fff',
                    }}
                  />
                </a>
              )}
              {s.table
                ? <SourceTable table={s.table} />
                : <pre style={{ whiteSpace: 'pre-wrap' }}>{s.text}</pre>}
            </details>
          ))}
        </section>
      )}
    </main>
  )
}