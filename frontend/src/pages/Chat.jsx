import { useState } from 'react'

const SUGGESTIONS = [
  "What was Apple's total net sales in 2024?",
  "What was Microsoft's net income in fiscal 2025?",
  "Why did Apple's Services gross margin change?",
  "What was Alphabet's revenue in 2024?",
]

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

  // turn "[Source 2]" / "[2]" / "[Sources 1, 3]" into clickable superscripts
  function renderAnswer(text) {
    const re = /\[(?:Sources?\s*)?(\d+(?:\s*,\s*\d+)*)\]/g
    const out = []
    let last = 0, m
    while ((m = re.exec(text)) !== null) {
      out.push(text.slice(last, m.index))
      m[1].split(',').map(s => s.trim()).forEach((id, i) => {
        out.push(
          <sup key={`${m.index}-${i}`}>
            <button className="cite" onClick={() => openSource(Number(id))}>[{id}]</button>
          </sup>
        )
      })
      last = m.index + m[0].length
    }
    out.push(text.slice(last))
    return out
  }

  const ver = res?.verification || []

  return (
    <main className="ask-wrap">
      <h1 className="ask-title">FilingLens</h1>
      <p className="ask-sub">
        Ask about Apple and Microsoft 10-K filings. Every number in the answer is
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
          <div className="result-card">{renderAnswer(res.answer)}</div>
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
              <pre>{s.text}</pre>
            </details>
          ))}
        </section>
      )}
    </main>
  )
}