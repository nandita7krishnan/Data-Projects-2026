"""
Topic modeling with LDA (scikit-learn).

Discovers recurring discussion themes across all collected posts/comments
and assigns a dominant topic to each piece of content.
"""

import logging
import re
import sqlite3

import numpy as np
import pandas as pd
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config.settings import DB_PATH, PROCESSED_DIR, N_TOPICS, N_TOP_WORDS

logger = logging.getLogger(__name__)

# Download NLTK assets quietly on first run
for _resource, _kind in [("stopwords", "corpora"), ("punkt_tab", "tokenizers")]:
    try:
        nltk.data.find(f"{_kind}/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords  # noqa: E402 — must follow download

# Extra stopwords specific to AI Reddit posts
_AI_STOPWORDS = {
    "ai", "artificial", "intelligence", "machine", "learning", "model",
    "like", "just", "think", "know", "want", "use", "get", "one",
    "people", "way", "really", "thing", "make", "time", "would",
    "could", "also", "even", "much", "many", "well", "good", "new",
    "said", "going", "say", "see", "need", "come", "actually",
}


def _clean(text: str) -> str:
    """Strip URLs, non-alpha chars, lowercase."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()


class TopicAnalyzer:
    def __init__(self, n_topics: int = N_TOPICS, db_path=DB_PATH):
        self.n_topics = n_topics
        self.db_path = db_path
        self.vectorizer: CountVectorizer | None = None
        self.lda: LatentDirichletAllocation | None = None
        self._doc_df: pd.DataFrame | None = None
        self._dtm = None

    # --- internal ----------------------------------------------------------

    def _load_texts(self) -> tuple[pd.DataFrame, list[str]]:
        conn = sqlite3.connect(self.db_path)
        posts = pd.read_sql("SELECT id, subreddit, title, body FROM posts", conn)
        comments = pd.read_sql("SELECT id, subreddit, body FROM comments", conn)
        conn.close()

        posts["text"] = (
            posts["title"].fillna("") + " " + posts["body"].fillna("")
        ).apply(_clean)
        comments["text"] = comments["body"].fillna("").apply(_clean)

        df = pd.concat(
            [posts[["id", "subreddit", "text"]], comments[["id", "subreddit", "text"]]],
            ignore_index=True,
        )
        return df, df["text"].tolist()

    # --- public API --------------------------------------------------------

    def fit(self) -> None:
        """Fit LDA on all collected content."""
        self._doc_df, texts = self._load_texts()

        stop_words = list(set(stopwords.words("english")) | _AI_STOPWORDS)

        self.vectorizer = CountVectorizer(
            max_df=0.90,
            min_df=5,
            max_features=3000,
            stop_words=stop_words,
            ngram_range=(1, 2),
        )
        self._dtm = self.vectorizer.fit_transform(texts)
        self._feature_names = self.vectorizer.get_feature_names_out()

        logger.info(
            "Fitting LDA: %d topics on %d documents (%d features)…",
            self.n_topics, self._dtm.shape[0], self._dtm.shape[1],
        )
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
            learning_method="online",
        )
        self.lda.fit(self._dtm)
        logger.info("LDA fitting complete.")

    def get_topic_keywords(self, n_words: int = N_TOP_WORDS) -> list[dict]:
        """Return top keywords for each topic."""
        if self.lda is None:
            raise RuntimeError("Call fit() first.")
        topics = []
        for idx, component in enumerate(self.lda.components_):
            top_idx = component.argsort()[: -n_words - 1 : -1]
            keywords = [self._feature_names[i] for i in top_idx]
            topics.append({
                "topic_id":    idx,
                "label":       f"Topic {idx + 1}",
                "keywords":    keywords,
                "keywords_str": ", ".join(keywords),
            })
        return topics

    def assign_topics(self) -> pd.DataFrame:
        """Assign dominant topic to every document and save to CSV."""
        if self.lda is None:
            raise RuntimeError("Call fit() first.")

        topic_dist = self.lda.transform(self._dtm)
        df = self._doc_df.copy()
        df["topic_id"] = np.argmax(topic_dist, axis=1)
        df["topic_confidence"] = topic_dist.max(axis=1).round(4)

        kw_map = {t["topic_id"]: t["keywords_str"] for t in self.get_topic_keywords()}
        df["topic_keywords"] = df["topic_id"].map(kw_map)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / "topic_assignments.csv", index=False)
        return df

    def topic_distribution_by_subreddit(self) -> pd.DataFrame:
        """Return a pivot table: subreddit × topic counts."""
        df = self.assign_topics()
        pivot = df.groupby(["subreddit", "topic_id"]).size().unstack(fill_value=0)
        pivot.columns = [f"Topic {i + 1}" for i in pivot.columns]
        return pivot
