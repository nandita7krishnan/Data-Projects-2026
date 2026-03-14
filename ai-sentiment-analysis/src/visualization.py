"""
Visualization module — static (matplotlib/seaborn) and interactive (plotly) charts.

Static charts are saved to data/plots/.
Plotly functions return Figure objects for use in Streamlit.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud

from config.settings import PLOTS_DIR, SENTIMENT_COLORS, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> str:
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart → %s", path)
    return str(path)


def _ordered_cols(df_cols) -> list[str]:
    """Return SENTIMENT_LABELS in order, keeping only those present in df."""
    return [lbl for lbl in SENTIMENT_LABELS if lbl in df_cols]


# ---------------------------------------------------------------------------
# Static charts (matplotlib / seaborn)
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(df: pd.DataFrame, save: bool = True):
    """Stacked bar chart: sentiment breakdown per subreddit."""
    counts = (
        df.groupby(["subreddit", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )
    cols = _ordered_cols(counts.columns)
    counts = counts[cols]
    colors = [SENTIMENT_COLORS.get(c, "#aaa") for c in cols]

    ax = counts.plot(kind="bar", stacked=True, figsize=(14, 6),
                     color=colors, edgecolor="white")
    ax.set_title("Sentiment Distribution by Subreddit", fontsize=14, fontweight="bold")
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("Posts / Comments")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save:
        return _save(ax.figure, "sentiment_distribution.png")
    return ax.figure


def plot_sentiment_trend(df: pd.DataFrame, freq: str = "W", save: bool = True):
    """Line chart: sentiment volume over time."""
    df = df.copy()
    df["period"] = df["created_dt"].dt.to_period(freq).dt.to_timestamp()

    trend = (
        df.groupby(["period", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )
    cols = _ordered_cols(trend.columns)
    trend = trend[cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    for col in cols:
        ax.plot(trend.index, trend[col], marker="o", markersize=3,
                label=col, color=SENTIMENT_COLORS.get(col, "#aaa"))
    ax.set_title("Sentiment Trends Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        return _save(fig, "sentiment_trend.png")
    return fig


def plot_subreddit_heatmap(df: pd.DataFrame, save: bool = True):
    """Heatmap: normalised sentiment share per subreddit."""
    counts = df.groupby(["subreddit", "sentiment_label"]).size().unstack(fill_value=0)
    norm = counts.div(counts.sum(axis=1), axis=0)
    cols = _ordered_cols(norm.columns)
    norm = norm[cols]

    fig, ax = plt.subplots(figsize=(13, max(4, len(norm) * 0.7)))
    sns.heatmap(norm, annot=True, fmt=".0%", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Share"})
    ax.set_title("Sentiment Share per Subreddit (normalised)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    if save:
        return _save(fig, "sentiment_heatmap.png")
    return fig


def plot_wordclouds(df: pd.DataFrame, save: bool = True) -> list[str]:
    """One word cloud per sentiment category."""
    paths = []
    for label in SENTIMENT_LABELS:
        texts = df[df["sentiment_label"] == label]["text"].dropna()
        if texts.empty:
            continue

        combined = " ".join(texts.astype(str))
        hex_color = SENTIMENT_COLORS.get(label, "#333333").lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        wc = WordCloud(
            width=900, height=450,
            background_color="white",
            color_func=lambda *a, **kw: f"rgb({r},{g},{b})",
            max_words=80,
            collocations=False,
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {label}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            fname = f"wordcloud_{label.replace(' ', '_').lower()}.png"
            paths.append(_save(fig, fname))
    return paths


# ---------------------------------------------------------------------------
# Interactive Plotly charts (Streamlit dashboard)
# ---------------------------------------------------------------------------

def plotly_sentiment_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart (interactive)."""
    counts = (
        df.groupby(["subreddit", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        counts, x="subreddit", y="count", color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution by Subreddit",
        labels={"count": "Posts / Comments", "sentiment_label": "Sentiment"},
        barmode="stack",
        category_orders={"sentiment_label": SENTIMENT_LABELS},
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.55))
    return fig


def plotly_sentiment_trend(df: pd.DataFrame, freq: str = "W") -> go.Figure:
    """Line chart — sentiment over time (interactive)."""
    df = df.copy()
    df["period"] = df["created_dt"].dt.to_period(freq).dt.to_timestamp()
    trend = (
        df.groupby(["period", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    fig = px.line(
        trend, x="period", y="count", color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Trends Over Time",
        labels={"period": "Date", "count": "Count", "sentiment_label": "Sentiment"},
        markers=True,
        category_orders={"sentiment_label": SENTIMENT_LABELS},
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.45))
    return fig


def plotly_pie(df: pd.DataFrame, subreddit: str) -> go.Figure:
    """Donut chart for a single subreddit."""
    sub_df = df[df["subreddit"] == subreddit]
    counts = sub_df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    fig = px.pie(
        counts, values="count", names="label",
        color="label", color_discrete_map=SENTIMENT_COLORS,
        title=f"Sentiment Breakdown — r/{subreddit}",
        hole=0.38,
    )
    return fig


def plotly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Normalised sentiment heatmap (interactive)."""
    counts = df.groupby(["subreddit", "sentiment_label"]).size().unstack(fill_value=0)
    norm = counts.div(counts.sum(axis=1), axis=0).round(3)
    cols = _ordered_cols(norm.columns)
    norm = norm[cols]

    fig = px.imshow(
        norm,
        text_auto=".0%",
        color_continuous_scale="RdYlGn",
        title="Sentiment Share per Subreddit",
        labels={"x": "Sentiment", "y": "Subreddit", "color": "Share"},
    )
    fig.update_xaxes(tickangle=30)
    return fig
