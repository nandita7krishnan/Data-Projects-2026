import { useState } from 'react'

function PitchInput({ onSubmit, disabled }) {
  const [pitch, setPitch] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (pitch.trim() && !disabled) {
      onSubmit(pitch.trim())
    }
  }

  return (
    <form className="pitch-input" onSubmit={handleSubmit}>
      <textarea
        value={pitch}
        onChange={(e) => setPitch(e.target.value)}
        placeholder="Pitch your idea to the boardroom..."
        rows={3}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !pitch.trim()}>
        {disabled ? 'Debating...' : 'Submit Pitch'}
      </button>
    </form>
  )
}

export default PitchInput
