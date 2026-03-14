"""
Streamlit interactive dashboard for AI Reddit Sentiment Analysis.

Run with:
    streamlit run dashboard/app.py
"""

import sqlite3
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from config.settings import DB_PATH, SENTIMENT_COLORS, SENTIMENT_LABELS
from src.visualization import (
    plotly_heatmap,
    plotly_pie,
    plotly_sentiment_bar,
    plotly_sentiment_trend,
)

st.set_page_config(
    page_title="AI Reddit Sentiment",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df_all: pd.DataFrame):
    st.sidebar.header("Filters")

    available_subs = sorted(df_all["subreddit"].unique().tolist())
    selected_subs = st.sidebar.multiselect(
        "Subreddits", available_subs, default=available_subs
    )

    content_types = st.sidebar.multiselect(
        "Content Type", ["post", "comment"], default=["post", "comment"]
    )

    date_min = df_all["created_dt"].min().date()
    date_max = df_all["created_dt"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    freq = st.sidebar.selectbox(
        "Trend Granularity",
        ["D", "W", "M"],
        index=1,
        format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x],
    )

    only_analyzed = st.sidebar.checkbox("Only show analysed content", value=True)

    return selected_subs, content_types, date_range, freq, only_analyzed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🤖 AI Reddit Sentiment Analyser")
    st.caption("Explore how Reddit communities feel about AI.")

    df_all = load_data()

    if df_all.empty:
        st.warning("No data found. Run the pipeline first:")
        st.code(
            "python main.py collect --subreddits womenintech --limit 100\n"
            "python main.py analyze"
        )
        return

    selected_subs, content_types, date_range, freq, only_analyzed = render_sidebar(df_all)

    # Apply filters
    df = df_all.copy()
    if selected_subs:
        df = df[df["subreddit"].isin(selected_subs)]
    if content_types:
        df = df[df["content_type"].isin(content_types)]
    if len(date_range) == 2:
        df = df[
            (df["created_dt"].dt.date >= date_range[0])
            & (df["created_dt"].dt.date <= date_range[1])
        ]
    if only_analyzed:
        df = df[df["sentiment_label"].notna()]

    if df.empty:
        st.info("No records match the current filters.")
        return

    # --- KPI row -----------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", f"{len(df):,}")
    col2.metric("Subreddits", len(df["subreddit"].unique()))
    col3.metric("Analysed", f"{df['sentiment_label'].notna().sum():,}")

    if df["sentiment_label"].notna().any():
        dominant = df["sentiment_label"].value_counts().idxmax()
        col4.metric("Dominant Sentiment", dominant)

    st.divider()

    # --- Tabs --------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distribution",
        "📈 Trends",
        "🗺️ Heatmap",
        "🍩 Per Subreddit",
        "📝 Raw Data",
    ])

    with tab1:
        st.plotly_chart(plotly_sentiment_bar(df), use_container_width=True)

    with tab2:
        st.plotly_chart(plotly_sentiment_trend(df, freq=freq), use_container_width=True)

    with tab3:
        st.plotly_chart(plotly_heatmap(df), use_container_width=True)

    with tab4:
        subs_with_data = sorted(df["subreddit"].unique().tolist())
        if subs_with_data:
            sub_choice = st.selectbox("Select subreddit", subs_with_data)
            st.plotly_chart(plotly_pie(df, sub_choice), use_container_width=True)

            st.subheader(f"Top Posts — r/{sub_choice}")
            top_posts = (
                df[
                    (df["subreddit"] == sub_choice)
                    & (df["content_type"] == "post")
                ]
                .sort_values("score", ascending=False)
                .head(10)[["text", "score", "sentiment_label", "sentiment_score", "created_dt"]]
            )
            st.dataframe(top_posts, use_container_width=True)

    with tab5:
        st.dataframe(
            df[
                ["subreddit", "content_type", "text", "score",
                 "sentiment_label", "sentiment_score", "created_dt"]
            ]
            .sort_values("created_dt", ascending=False)
            .head(500),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
