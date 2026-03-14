# AI Reddit Sentiment Analyzer

Tracks how Reddit communities feel about AI over time — optimistic, fearful, critical, excited, and more.

**Starting focus:** r/womenintech — then expanding to r/artificial, r/MachineLearning, r/singularity, r/ChatGPT, r/technology, r/LocalLLaMA, r/OpenAI.

---

## Project Structure

```
ai-sentiment-analysis/
├── config/
│   └── settings.py          # Subreddits, sentiment labels, paths, model config
├── data/
│   ├── raw/                 # Exported CSVs (posts.csv, comments.csv)
│   ├── processed/           # Sentiment-tagged CSVs, topic assignments
│   ├── plots/               # Generated chart images
│   └── reddit_sentiment.db  # SQLite database
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
├── src/
│   ├── reddit_scraper.py    # PRAW collection + SQLite storage
│   ├── sentiment_analysis.py# HuggingFace zero-shot classification
│   ├── topic_analysis.py    # LDA topic modeling
│   └── visualization.py     # matplotlib, seaborn, plotly charts
├── main.py                  # CLI entrypoint
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get Reddit API credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click **"create another app"**
3. Choose **script**
4. Note your **client_id** (under the app name) and **client_secret**

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=ai_sentiment_bot/1.0 by u/your_username
```

---

## Running the Pipeline

### Step 1 — Collect data

```bash
# Start with r/womenintech
python main.py collect --subreddits womenintech --limit 100 --time-filter month

# Add more subreddits later
python main.py collect --subreddits artificial MachineLearning singularity --limit 100

# Collect from all configured subreddits
python main.py collect --limit 200 --time-filter year
```

### Step 2 — Run sentiment analysis

```bash
python main.py analyze
# or just posts/comments:
python main.py analyze --content posts
```

> **Note:** First run downloads the `facebook/bart-large-mnli` model (~1.6 GB). Subsequent runs use the cached model.

### Step 3 — Topic modeling (optional)

```bash
python main.py topics --n-topics 8
```

### Step 4 — Generate static charts

```bash
python main.py visualize
# Charts saved to data/plots/
```

### Step 5 — Interactive dashboard

```bash
streamlit run dashboard/app.py
```

### Check database status

```bash
python main.py status
```

---

## Sentiment Categories

| Label | Meaning |
|---|---|
| Optimistic about AI | Positive outlook on AI's future |
| Excited about AI capabilities | Enthusiasm about current AI features |
| Neutral or Informational | Factual, no strong opinion |
| Critical of AI hype | Skeptical of overblown claims |
| Concerned about AI | Worried about risks or impact |
| Fearful of AI | Strong negative/existential concern |

---

## Dashboard Features

- **Subreddit selector** — compare any combination
- **Date range filter** — zoom into specific periods
- **Content type filter** — posts, comments, or both
- **Trend granularity** — daily / weekly / monthly
- **5 tabs:** Distribution · Trends · Heatmap · Per-Subreddit Donut · Raw Data

---

## Extending the Project

- **Add subreddits:** Edit `SUBREDDITS` in `config/settings.py`
- **Change sentiment labels:** Edit `SENTIMENT_LABELS` in `config/settings.py`
- **Use a faster model:** Replace `ZERO_SHOT_MODEL` with `"cross-encoder/nli-deberta-v3-small"` for ~5x speed
- **GPU inference:** Set `device=0` in `SentimentAnalyzer.classifier`
