import { useState } from 'react'
import PitchInput from './components/PitchInput'
import DebateThread from './components/DebateThread'
import VoteTally from './components/VoteTally'

function App() {
  const [debate, setDebate] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmitPitch = async (pitch) => {
    setLoading(true)
    setError(null)
    setDebate(null)

    try {
      const response = await fetch('/pitch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pitch }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setDebate(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>The Boardroom</h1>
        <p className="subtitle">Pitch an idea. Six characters debate it.</p>
      </header>

      <PitchInput onSubmit={handleSubmitPitch} disabled={loading} />

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>The boardroom is deliberating... This may take a minute.</p>
        </div>
      )}

      {error && (
        <div className="error">
          <p>Something went wrong: {error}</p>
        </div>
      )}

      {debate && (
        <>
          <DebateThread phases={debate.phases} pitch={debate.pitch} />
          <VoteTally votes={debate.votes} />
        </>
      )}
    </div>
  )
}

export default App
