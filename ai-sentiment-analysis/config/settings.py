from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = DATA_DIR / "plots"

# Database
DB_PATH = DATA_DIR / "reddit_sentiment.db"

# Reddit — start with womenintech, expand later
SUBREDDITS = [
    "womenintech",
    "artificial",
    "MachineLearning",
    "singularity",
    "ChatGPT",
    "technology",
    "LocalLLaMA",
    "OpenAI",
]

DEFAULT_POST_LIMIT = 100
TOP_COMMENTS_PER_POST = 5
TIME_FILTERS = ["day", "week", "month", "year", "all"]

# Sentiment categories
SENTIMENT_LABELS = [
    "Optimistic about AI",
    "Excited about AI capabilities",
    "Neutral or Informational",
    "Critical of AI hype",
    "Concerned about AI",
    "Fearful of AI",
]

SENTIMENT_COLORS = {
    "Optimistic about AI": "#2ecc71",
    "Excited about AI capabilities": "#27ae60",
    "Neutral or Informational": "#95a5a6",
    "Critical of AI hype": "#e67e22",
    "Concerned about AI": "#e74c3c",
    "Fearful of AI": "#8e44ad",
}

# Model
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
BATCH_SIZE = 8
MAX_TEXT_LENGTH = 512

# Topic modeling
N_TOPICS = 8
N_TOP_WORDS = 10
