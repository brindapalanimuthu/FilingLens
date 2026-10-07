import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <main className="wrap">
      <h1>Ask SEC filings. Trust every number.</h1>
      <p className="lead">
        FilingLens answers questions over Apple and Microsoft 10-Ks and checks
        every figure against the source text before showing it.
      </p>
      <Link to="/ask" className="cta">Try it</Link>

      <section className="grid">
        <div className="card"><h3>Cited answers</h3><p>Every claim links to the filing passage it came from.</p></div>
        <div className="card"><h3>Number check</h3><p>Figures in the answer are verified against retrieved sources.</p></div>
        <div className="card"><h3>Smart routing</h3><p>Numeric questions hit tables, "why" questions hit text.</p></div>
      </section>
    </main>
  )
}