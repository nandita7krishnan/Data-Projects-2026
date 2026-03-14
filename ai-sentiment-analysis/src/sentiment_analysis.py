"""
Sentiment analysis using HuggingFace zero-shot classification.

Classifies each post/comment into one of the AI-specific sentiment
categories defined in config.settings.SENTIMENT_LABELS.

The model is lazy-loaded on first use so imports stay fast.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from config.settings import (
    DB_PATH, PROCESSED_DIR,
    SENTIMENT_LABELS, ZERO_SHOT_MODEL, BATCH_SIZE, MAX_TEXT_LENGTH,
)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, model_name: str = ZERO_SHOT_MODEL, db_path=DB_PATH):
        self.model_name = model_name
        self.db_path = db_path
        self._classifier = None  # lazy-loaded

    # --- model -------------------------------------------------------------

    @property
    def classifier(self):
        if self._classifier is None:
            logger.info(
                "Loading zero-shot model: %s  (first run downloads ~1.6 GB)",
                self.model_name,
            )
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,          # CPU; set to 0 to use GPU
            )
        return self._classifier

    # --- text prep ---------------------------------------------------------

    @staticmethod
    def _build_text(row: pd.Series, content_type: str) -> str:
        """Combine title + body for posts; use body for comments."""
        if content_type == "post":
            text = f"{row.get('title', '')} {row.get('body', '')}".strip()
        else:
            text = str(row.get("body", "")).strip()
        # Rough truncation (4 chars ≈ 1 token)
        return text[: MAX_TEXT_LENGTH * 4] or "no content"

    # --- inference ---------------------------------------------------------

    def analyze_text(self, text: str) -> dict:
        """Classify a single string. Returns {label, score}."""
        text = text.strip() or "no content"
        result = self.classifier(text, SENTIMENT_LABELS, multi_label=False)
        return {
            "label": result["labels"][0],
            "score": round(result["scores"][0], 4),
        }

    def _classify_batch(self, texts: list[str]) -> list[dict]:
        """Run zero-shot classification in batches with a progress bar."""
        results = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Classifying"):
            batch = [t or "no content" for t in texts[i : i + BATCH_SIZE]]
            outputs = self.classifier(batch, SENTIMENT_LABELS, multi_label=False)
            # outputs is a list when input is a list
            if isinstance(outputs, dict):
                outputs = [outputs]
            for out in outputs:
                results.append({
                    "label": out["labels"][0],
                    "score": round(out["scores"][0], 4),
                })
        return results

    # --- DataFrame-level API -----------------------------------------------

    def analyze_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = [self._build_text(row, "post") for _, row in df.iterrows()]
        results = self._classify_batch(texts)
        df = df.copy()
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
        return df

    def analyze_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = [self._build_text(row, "comment") for _, row in df.iterrows()]
        results = self._classify_batch(texts)
        df = df.copy()
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
        return df

    # --- persistence -------------------------------------------------------

    def save_results(self, df: pd.DataFrame, content_type: str) -> None:
        """Write sentiment results to SQLite and CSV."""
        now = datetime.utcnow().isoformat()
        records = [
            {
                "content_id":      row["id"],
                "content_type":    content_type,
                "sentiment_label": row["sentiment_label"],
                "sentiment_score": row["sentiment_score"],
                "analyzed_at":     now,
            }
            for _, row in df.iterrows()
        ]

        conn = sqlite3.connect(self.db_path)
        # INSERT OR REPLACE to allow re-analysis
        pd.DataFrame(records).to_sql(
            "sentiment_results", conn, if_exists="replace", index=False
        )
        conn.close()

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / f"{content_type}s_with_sentiment.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %d %s sentiment results → %s", len(records), content_type, path)

    def load_results(self, content_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load sentiment-enriched data from SQLite.

        content_type: 'post' | 'comment' | None (returns both combined)
        """
        conn = sqlite3.connect(self.db_path)

        if content_type == "post":
            query = """
                SELECT p.id, p.subreddit,
                       (p.title || ' ' || COALESCE(p.body, '')) AS text,
                       p.score, p.created_utc, 'post' AS content_type,
                       s.sentiment_label, s.sentiment_score
                FROM posts p
                LEFT JOIN sentiment_results s
                       ON p.id = s.content_id AND s.content_type = 'post'
            """
        elif content_type == "comment":
            query = """
                SELECT c.id, c.subreddit, c.body AS text,
                       c.score, c.created_utc, 'comment' AS content_type,
                       s.sentiment_label, s.sentiment_score
                FROM comments c
                LEFT JOIN sentiment_results s
                       ON c.id = s.content_id AND s.content_type = 'comment'
            """
        else:
            query = """
                SELECT p.id, p.subreddit,
                       (p.title || ' ' || COALESCE(p.body, '')) AS text,
                       p.score, p.created_utc, 'post' AS content_type,
                       s.sentiment_label, s.sentiment_score
                FROM posts p
                LEFT JOIN sentiment_results s
                       ON p.id = s.content_id AND s.content_type = 'post'
                UNION ALL
                SELECT c.id, c.subreddit, c.body AS text,
                       c.score, c.created_utc, 'comment' AS content_type,
                       s.sentiment_label, s.sentiment_score
                FROM comments c
                LEFT JOIN sentiment_results s
                       ON c.id = s.content_id AND s.content_type = 'comment'
            """

        df = pd.read_sql(query, conn)
        conn.close()
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        return df
