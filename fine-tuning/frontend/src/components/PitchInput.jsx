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
        placeholder="Present your proposal to the board..."
        rows={3}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !pitch.trim()}>
        {disabled ? 'Deliberating...' : 'Present to the Board'}
      </button>
    </form>
  )
}

export default PitchInput
