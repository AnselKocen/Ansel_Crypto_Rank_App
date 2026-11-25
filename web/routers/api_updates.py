"""Shared update status, idempotent requests, and versioned published assets."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from web.services.weekly_updates import get_updates

router = APIRouter(prefix="/updates", tags=["weekly-updates"])
FILES = {"wordcloud_positive.png": "figures/wordcloud_positive.png",
         "wordcloud_negative.png": "figures/wordcloud_negative.png",
         "fear_greed_gauge.png": "figures/fear_greed_gauge.png",
         "history.csv": "df_merged_history.csv", "coverage.csv": "daily_coverage.csv",
         "manifest.json": "manifest.json"}


class UpdateRequest(BaseModel):
    component: str = "all"


@router.get("/status")
def status(component: str = "all"):
    if component not in ("all", "data", "report"):
        raise HTTPException(400, "Unknown update component")
    return get_updates().status(component)


@router.post("/run")
def request_update(request: UpdateRequest):
    if request.component not in ("all", "data", "report"):
        raise HTTPException(400, "Unknown update component")
    return get_updates().request("manual", request.component)


@router.get("/versions/{version}/{filename}")
def asset(version: str, filename: str):
    service = get_updates()
    directory = (service.root / "versions" / version).resolve()
    if directory.parent != service.root / "versions" or filename not in FILES:
        raise HTTPException(404, "Unknown published file")
    path = directory / FILES[filename]
    if not (directory / "manifest.json").is_file() or not path.is_file():
        raise HTTPException(404, "File not available")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})
