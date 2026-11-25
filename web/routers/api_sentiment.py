"""Sentiment reads a published version and updates through the shared job."""
from fastapi import APIRouter
from web.services.sentiment_service import get_wordcloud_paths, get_fear_greed_path
from web.services.weekly_updates import get_updates

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.get("/status")
def get_status():
    s = get_updates().status("data")
    return {**s, "last_updated": s["data_period"], "is_outdated": not s["data_current"],
            "exists": s["data_period"] is not None, "update_running": s["running"],
            "update_error": (s["last_job"] or {}).get("error")}


@router.get("/wordcloud")
def get_wordcloud():
    return get_wordcloud_paths()


@router.get("/fear-greed")
def get_fear_greed():
    return get_fear_greed_path()


@router.post("/update")
def trigger_update():
    return get_updates().request("sentiment", "data")
