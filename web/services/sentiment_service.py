"""
Sentiment Analysis Service
Wraps existing sentiment_update.py functionality
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

FIGURES_DIR = BASE_DIR / "figures"
UPDATE_FILE = BASE_DIR / "last_updated_wordcloud.txt"


def get_last_update() -> dict:
    """Get the last update timestamp for sentiment data"""
    if UPDATE_FILE.exists():
        last_updated_str = UPDATE_FILE.read_text().strip()
        try:
            last_updated_date = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            is_outdated = last_updated_date < today
            return {
                "last_updated": last_updated_str,
                "is_outdated": is_outdated,
                "exists": True
            }
        except ValueError:
            return {
                "last_updated": None,
                "is_outdated": True,
                "exists": False,
                "error": "Invalid date format"
            }
    return {
        "last_updated": None,
        "is_outdated": True,
        "exists": False
    }


def _published_image(filename):
    from web.services.weekly_updates import get_updates, data_path
    publication = get_updates().published()
    path = data_path("figures/"+filename)
    url = (f"/api/updates/versions/{publication['path'].name}/{filename}" if publication
           else "/figures/"+filename)
    return {"path": str(path), "exists": path.is_file(), "url": url}


def get_wordcloud_paths():
    return {"positive": _published_image("wordcloud_positive.png"),
            "negative": _published_image("wordcloud_negative.png")}


def get_fear_greed_path():
    return _published_image("fear_greed_gauge.png")


async def run_sentiment_update():
    from web.services.weekly_updates import get_updates
    return get_updates().request("sentiment", "data")
