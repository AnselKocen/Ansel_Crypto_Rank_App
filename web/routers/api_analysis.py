"""AI reports use one shared weekly generation job across visitors."""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel
from web.services.analysis_service import get_report_content
from web.services.weekly_updates import get_updates

router = APIRouter(prefix="/analysis", tags=["analysis"])


class GenerateRequest(BaseModel):
    force: bool = False  # Accepted for compatibility; never bypasses the cache.


@router.get("/status")
def get_status():
    state = get_updates().status("report")
    return {**state, "should_update": not state["report_current"],
            "last_update": state["data_period"] if state["report_current"] else None,
            "generation_running": state["running"],
            "generation_error": state["report_error"]}


@router.get("/report")
def get_report():
    content = get_report_content()
    state = get_updates().status("report")
    return {"content": content, "exists": content is not None,
            "is_current": state["report_current"], "data_period": state["data_period"]}


@router.post("/generate")
def trigger_generate(request: GenerateRequest):
    return get_updates().request("report", "report")


@router.get("/generation-result")
def get_generation_result():
    state = get_updates().status("report")
    return {"running": state["running"], "error": state["report_error"],
            "result": {"content": get_report_content(), "message": "本周共享报告已就绪"}
                      if state["report_current"] else None}


@router.get("/download")
def download_report():
    content = get_report_content()
    if content is None:
        return PlainTextResponse("No report available", status_code=404)
    return Response(content, media_type="text/markdown",
                    headers={"Content-Disposition": 'attachment; filename="crypto_market_report.md"'})
