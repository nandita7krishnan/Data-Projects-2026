"""
Reddit data collection using PRAW.

Fetches posts and top comments from specified subreddits and stores
them in SQLite. Re-running skips already-collected post IDs.
"""

import os
import logging
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd
import praw
from dotenv import load_dotenv

from config.settings import (
    DB_PATH, RAW_DIR, SUBREDDITS,
    DEFAULT_POST_LIMIT, TOP_COMMENTS_PER_POST,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def init_database(db_path=DB_PATH) -> None:
    """Create tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id            TEXT PRIMARY KEY,
            subreddit     TEXT NOT NULL,
            title         TEXT,
            body          TEXT,
            score         INTEGER,
            upvote_ratio  REAL,
            num_comments  INTEGER,
            created_utc   INTEGER,
            author        TEXT,
            url           TEXT,
            collected_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS comments (
            id           TEXT PRIMARY KEY,
            post_id      TEXT NOT NULL,
            subreddit    TEXT NOT NULL,
            body         TEXT,
            score        INTEGER,
            created_utc  INTEGER,
            author       TEXT,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        );

        CREATE TABLE IF NOT EXISTS sentiment_results (
            content_id    TEXT    NOT NULL,
            content_type  TEXT    NOT NULL,   -- 'post' or 'comment'
            sentiment_label TEXT,
            sentiment_score REAL,
            analyzed_at   TEXT,
            PRIMARY KEY (content_id, content_type)
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database ready at %s", db_path)


# ---------------------------------------------------------------------------
# Reddit client
# ---------------------------------------------------------------------------

def get_reddit_client() -> praw.Reddit:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "ai_sentiment_bot/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class RedditScraper:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.reddit = get_reddit_client()
        init_database(db_path)

    # --- helpers -----------------------------------------------------------

    def _existing_post_ids(self, subreddit: str) -> set:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT id FROM posts WHERE subreddit = ?", (subreddit,)
        )
        ids = {row[0] for row in cur.fetchall()}
        conn.close()
        return ids

    def _save_posts(self, records: list[dict]) -> None:
        conn = sqlite3.connect(self.db_path)
        pd.DataFrame(records).to_sql("posts", conn, if_exists="append", index=False)
        conn.close()

    def _save_comments(self, records: list[dict]) -> None:
        if not records:
            return
        conn = sqlite3.connect(self.db_path)
        existing = pd.read_sql("SELECT id FROM comments", conn)
        df = pd.DataFrame(records)
        df = df[~df["id"].isin(existing["id"])]
        if not df.empty:
            df.to_sql("comments", conn, if_exists="append", index=False)
        conn.close()

    # --- collection --------------------------------------------------------

    def collect_subreddit(
        self,
        subreddit_name: str,
        limit: int = DEFAULT_POST_LIMIT,
        time_filter: str = "month",
    ) -> tuple[int, int]:
        """
        Collect top posts + comments from one subreddit.
        Returns (posts_added, comments_added).
        """
        logger.info(
            "Collecting r/%s  limit=%d  time_filter=%s",
            subreddit_name, limit, time_filter,
        )
        existing_ids = self._existing_post_ids(subreddit_name)
        subreddit = self.reddit.subreddit(subreddit_name)

        posts_data: list[dict] = []
        comments_data: list[dict] = []

        for submission in subreddit.top(time_filter=time_filter, limit=limit):
            if submission.id in existing_ids:
                continue

            posts_data.append({
                "id":           submission.id,
                "subreddit":    subreddit_name,
                "title":        submission.title,
                "body":         submission.selftext or "",
                "score":        submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "created_utc":  int(submission.created_utc),
                "author":       str(submission.author) if submission.author else "[deleted]",
                "url":          submission.url,
                "collected_at": datetime.utcnow().isoformat(),
            })

            # Top comments only — avoid deep trees
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:TOP_COMMENTS_PER_POST]:
                body = getattr(comment, "body", "")
                if body in ("[deleted]", "[removed]", ""):
                    continue
                comments_data.append({
                    "id":          comment.id,
                    "post_id":     submission.id,
                    "subreddit":   subreddit_name,
                    "body":        body,
                    "score":       comment.score,
                    "created_utc": int(comment.created_utc),
                    "author":      str(comment.author) if comment.author else "[deleted]",
                })

        if posts_data:
            self._save_posts(posts_data)
        if comments_data:
            self._save_comments(comments_data)

        logger.info(
            "r/%s: +%d posts, +%d comments",
            subreddit_name, len(posts_data), len(comments_data),
        )
        return len(posts_data), len(comments_data)

    def collect_all(
        self,
        subreddits: list[str] = SUBREDDITS,
        limit: int = DEFAULT_POST_LIMIT,
        time_filter: str = "month",
    ) -> dict:
        """Collect from multiple subreddits. Returns per-subreddit summary."""
        results = {}
        for sub in subreddits:
            try:
                posts, comments = self.collect_subreddit(sub, limit, time_filter)
                results[sub] = {"posts": posts, "comments": comments}
            except Exception as exc:
                logger.error("Failed r/%s: %s", sub, exc)
                results[sub] = {"error": str(exc)}
        return results

    # --- loading -----------------------------------------------------------

    def load_posts(self, subreddits: Optional[list[str]] = None) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        if subreddits:
            ph = ",".join("?" * len(subreddits))
            df = pd.read_sql(
                f"SELECT * FROM posts WHERE subreddit IN ({ph})",
                conn, params=subreddits,
            )
        else:
            df = pd.read_sql("SELECT * FROM posts", conn)
        conn.close()
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        return df

    def load_comments(self, subreddits: Optional[list[str]] = None) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        if subreddits:
            ph = ",".join("?" * len(subreddits))
            df = pd.read_sql(
                f"SELECT * FROM comments WHERE subreddit IN ({ph})",
                conn, params=subreddits,
            )
        else:
            df = pd.read_sql("SELECT * FROM comments", conn)
        conn.close()
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        return df

    def export_csv(self, output_dir=RAW_DIR) -> None:
        """Dump posts and comments tables to CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        for table in ("posts", "comments"):
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            path = output_dir / f"{table}.csv"
            df.to_csv(path, index=False)
            logger.info("Exported %d rows → %s", len(df), path)
        conn.close()
