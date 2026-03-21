import { useState, useEffect } from 'react'
import CharacterMessage from './CharacterMessage'

function DebateThread({ phases, pitch }) {
  // Flatten all messages across phases into a single ordered list for staggered reveal
  const allMessages = []
  phases.forEach((phase, pi) => {
    phase.messages.forEach((msg, mi) => {
      allMessages.push({ ...msg, phaseIndex: pi, msgIndex: mi })
    })
  })

  const [visibleCount, setVisibleCount] = useState(0)
  const [animatingIndex, setAnimatingIndex] = useState(0)

  useEffect(() => {
    if (visibleCount >= allMessages.length) return

    // Delay before showing each new message
    const delay = visibleCount === 0 ? 600 : 2000
    const timer = setTimeout(() => {
      setVisibleCount((c) => c + 1)
      setAnimatingIndex(visibleCount)
    }, delay)

    return () => clearTimeout(timer)
  }, [visibleCount, allMessages.length])

  // Group visible messages back into phases for rendering
  const visibleMessages = allMessages.slice(0, visibleCount)

  return (
    <div className="debate-thread">
      <div className="pitch-display">
        <span className="pitch-label">THE PITCH</span>
        <p>{pitch}</p>
      </div>

      {phases.map((phase, pi) => {
        const phaseMessages = visibleMessages.filter((m) => m.phaseIndex === pi)
        if (phaseMessages.length === 0) return null

        return (
          <div key={pi} className="phase fade-in">
            <h2 className="phase-title">{phase.name}</h2>
            <div className="phase-messages">
              {phaseMessages.map((msg, j) => {
                // Find the global index for this message
                const globalIndex = allMessages.findIndex(
                  (m) => m.phaseIndex === msg.phaseIndex && m.msgIndex === msg.msgIndex
                )
                const shouldAnimate = globalIndex === animatingIndex

                return (
                  <div key={j} className="message-entrance">
                    <CharacterMessage
                      name={msg.name}
                      character={msg.character}
                      initials={msg.initials}
                      color={msg.color}
                      content={msg.content}
                      vote={msg.vote}
                      animate={shouldAnimate}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}

      {visibleCount < allMessages.length && (
        <div className="typing-indicator">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      )}
    </div>
  )
}

export default DebateThread
