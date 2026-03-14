"""
CLI entrypoint for the AI Reddit Sentiment Analysis pipeline.

Commands:
    python main.py collect   [--subreddits SUB ...] [--limit N] [--time-filter FILTER]
    python main.py analyze   [--content posts|comments|both]
    python main.py topics    [--n-topics N]
    python main.py visualize
    python main.py status
"""

import argparse
import logging
import sys

from config.settings import DB_PATH, SUBREDDITS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_collect(args):
    from src.reddit_scraper import RedditScraper

    scraper = RedditScraper()
    subs = args.subreddits or SUBREDDITS
    results = scraper.collect_all(subs, limit=args.limit, time_filter=args.time_filter)

    print("\n--- Collection Summary ---")
    total_posts = total_comments = 0
    for sub, counts in results.items():
        if "error" in counts:
            print(f"  r/{sub}: ERROR — {counts['error']}")
        else:
            print(f"  r/{sub}: +{counts['posts']} posts, +{counts['comments']} comments")
            total_posts += counts["posts"]
            total_comments += counts["comments"]
    print(f"\nTotal new: {total_posts} posts, {total_comments} comments")
    scraper.export_csv()


def cmd_analyze(args):
    from src.reddit_scraper import RedditScraper
    from src.sentiment_analysis import SentimentAnalyzer

    scraper = RedditScraper()
    analyzer = SentimentAnalyzer()

    if args.content in ("posts", "both"):
        posts_df = scraper.load_posts()
        if posts_df.empty:
            print("No posts found. Run 'collect' first.")
        else:
            print(f"Analyzing {len(posts_df)} posts…")
            posts_df = analyzer.analyze_posts(posts_df)
            analyzer.save_results(posts_df, "post")

    if args.content in ("comments", "both"):
        comments_df = scraper.load_comments()
        if comments_df.empty:
            print("No comments found. Run 'collect' first.")
        else:
            print(f"Analyzing {len(comments_df)} comments…")
            comments_df = analyzer.analyze_comments(comments_df)
            analyzer.save_results(comments_df, "comment")

    print("\nDone. Run 'python main.py visualize' to generate charts.")


def cmd_topics(args):
    from src.topic_analysis import TopicAnalyzer

    analyzer = TopicAnalyzer(n_topics=args.n_topics)
    analyzer.fit()

    print("\n--- Discovered Topics ---")
    for t in analyzer.get_topic_keywords():
        print(f"  Topic {t['topic_id'] + 1}: {t['keywords_str']}")

    dist = analyzer.topic_distribution_by_subreddit()
    print("\n--- Topic Distribution by Subreddit ---")
    print(dist.to_string())


def cmd_visualize(args):
    from src.sentiment_analysis import SentimentAnalyzer
    from src.visualization import (
        plot_sentiment_distribution,
        plot_sentiment_trend,
        plot_subreddit_heatmap,
        plot_wordclouds,
    )

    analyzer = SentimentAnalyzer()
    df = analyzer.load_results()

    if df["sentiment_label"].isna().all():
        print("No sentiment results found. Run 'python main.py analyze' first.")
        sys.exit(1)

    print("Generating charts…")
    plot_sentiment_distribution(df)
    plot_sentiment_trend(df)
    plot_subreddit_heatmap(df)
    plot_wordclouds(df)
    print("Charts saved to data/plots/")


def cmd_status(args):
    import sqlite3

    if not DB_PATH.exists():
        print("No database found. Run 'python main.py collect' first.")
        return

    conn = sqlite3.connect(DB_PATH)
    print("\n--- Database Status ---")
    for table in ("posts", "comments", "sentiment_results"):
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<20} {count:>6,} rows")
        except Exception:
            print(f"  {table:<20} (table not found)")
    conn.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Reddit Sentiment Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # collect
    p = sub.add_parser("collect", help="Scrape Reddit posts and comments")
    p.add_argument("--subreddits", nargs="+", default=None,
                   help="Override subreddit list (default: from config)")
    p.add_argument("--limit", type=int, default=100,
                   help="Max posts per subreddit (default: 100)")
    p.add_argument("--time-filter", default="month",
                   choices=["day", "week", "month", "year", "all"])
    p.set_defaults(func=cmd_collect)

    # analyze
    p = sub.add_parser("analyze", help="Run sentiment analysis")
    p.add_argument("--content", default="both",
                   choices=["posts", "comments", "both"])
    p.set_defaults(func=cmd_analyze)

    # topics
    p = sub.add_parser("topics", help="Run LDA topic modeling")
    p.add_argument("--n-topics", type=int, default=8)
    p.set_defaults(func=cmd_topics)

    # visualize
    p = sub.add_parser("visualize", help="Generate static charts")
    p.set_defaults(func=cmd_visualize)

    # status
    p = sub.add_parser("status", help="Show database row counts")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
