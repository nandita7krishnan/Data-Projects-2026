import { useState, useEffect } from 'react'

function VoteTally({ votes }) {
  const total = votes.yes + votes.no + votes.abstain
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), 300)
    return () => clearTimeout(timer)
  }, [])

  const getWidth = (count) => {
    if (!animated || total === 0) return '0%'
    return `${(count / total) * 100}%`
  }

  const verdict =
    votes.yes > votes.no ? 'APPROVED' : votes.no > votes.yes ? 'REJECTED' : 'SPLIT DECISION'

  const verdictClass =
    votes.yes > votes.no ? 'verdict-yes' : votes.no > votes.yes ? 'verdict-no' : 'verdict-split'

  return (
    <div className="vote-tally fade-in">
      <h2 className="phase-title">Vote Results</h2>
      <div className="vote-bars">
        <div className="vote-row">
          <span className="vote-label">YES</span>
          <div className="vote-bar-track">
            <div
              className="vote-bar vote-bar-yes"
              style={{ width: getWidth(votes.yes) }}
            />
          </div>
          <span className="vote-count">{votes.yes}</span>
        </div>
        <div className="vote-row">
          <span className="vote-label">NO</span>
          <div className="vote-bar-track">
            <div
              className="vote-bar vote-bar-no"
              style={{ width: getWidth(votes.no) }}
            />
          </div>
          <span className="vote-count">{votes.no}</span>
        </div>
        <div className="vote-row">
          <span className="vote-label">ABSTAIN</span>
          <div className="vote-bar-track">
            <div
              className="vote-bar vote-bar-abstain"
              style={{ width: getWidth(votes.abstain) }}
            />
          </div>
          <span className="vote-count">{votes.abstain}</span>
        </div>
      </div>
      <div className={`verdict ${verdictClass}`}>{verdict}</div>
    </div>
  )
}

export default VoteTally
