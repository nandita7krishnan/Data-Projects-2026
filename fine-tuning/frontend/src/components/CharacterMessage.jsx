import { useState, useEffect } from 'react'

function useTypewriter(text, speed = 18, enabled = true) {
  const [displayed, setDisplayed] = useState(enabled ? '' : text)
  const [done, setDone] = useState(!enabled)

  useEffect(() => {
    if (!enabled) {
      setDisplayed(text)
      setDone(true)
      return
    }

    setDisplayed('')
    setDone(false)
    let i = 0

    const interval = setInterval(() => {
      i++
      setDisplayed(text.slice(0, i))
      if (i >= text.length) {
        clearInterval(interval)
        setDone(true)
      }
    }, speed)

    return () => clearInterval(interval)
  }, [text, speed, enabled])

  return { displayed, done }
}

function CharacterMessage({ name, character, initials, color, content, vote, animate = true }) {
  const { displayed, done } = useTypewriter(content, 18, animate)
  const avatarSrc = `/avatars/${character}.png`
  const [imgError, setImgError] = useState(false)

  return (
    <div className="character-message">
      <div className="avatar-wrapper" style={{ '--char-color': color }}>
        {!imgError ? (
          <img
            src={avatarSrc}
            alt={name}
            className="avatar-img"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="avatar-fallback" style={{ backgroundColor: color }}>
            {initials}
          </div>
        )}
      </div>
      <div className="message-body">
        <div className="message-name" style={{ color }}>
          {name}
          {vote && (
            <span className={`vote-badge vote-${vote}`}>
              {vote.toUpperCase()}
            </span>
          )}
        </div>
        <div className="message-content">
          {displayed}
          {!done && <span className="typewriter-cursor">|</span>}
        </div>
      </div>
    </div>
  )
}

export default CharacterMessage
