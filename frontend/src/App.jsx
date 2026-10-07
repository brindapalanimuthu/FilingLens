import { Routes, Route, NavLink } from 'react-router-dom'
import Landing from './pages/Landing'
import Chat from './pages/Chat'

export default function App() {
  return (
    <>
      <nav className="nav">
        <NavLink to="/" className="logo">FilingLens</NavLink>
        <div>
          <NavLink to="/ask">Ask</NavLink>
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/ask" element={<Chat />} />
      </Routes>
    </>
  )
}