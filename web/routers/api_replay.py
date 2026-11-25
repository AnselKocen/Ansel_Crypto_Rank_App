"""Read-only access to archived historical replay results."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/replay", tags=["historical-replay"])
ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "historical_replay"
ALLOWED = {
    "forecast_history.csv", "forecast_actual_comparison.csv", "weekly_model_metrics.csv",
    "strategy_returns.csv", "strategy_summary.csv", "run_events.csv", "run_manifest.json",
}


def latest_directory():
    pointer = ROOT / "latest_run.json"
    if not pointer.exists():
        raise HTTPException(404, "Historical replay has not finished yet.")
    relative = json.loads(pointer.read_text())["path"]
    candidate = (ROOT / relative).resolve()
    if not candidate.is_relative_to((ROOT / "runs").resolve()):
        raise HTTPException(500, "Invalid replay path")
    return candidate


@router.get("/status")
def replay_status():
    pointer = ROOT / "latest_run.json"
    if not pointer.exists():
        return {"status": "preparing", "available": False}
    path = latest_directory() / "run_manifest.json"
    return {"available": True, **json.loads(path.read_text())}


@router.get("/download/{filename}")
def download_result(filename: str):
    if filename not in ALLOWED:
        raise HTTPException(404, "Unknown result file")
    path = latest_directory() / filename
    if not path.is_file():
        raise HTTPException(404, "Result not available")
    return FileResponse(path, filename=filename)


def experiment_directory():
    root = ROOT / "target_experiment"
    pointer = root / "latest_run.json"
    if not pointer.is_file():
        raise HTTPException(404, "Target experiment has not finished yet.")
    candidate = (root / json.loads(pointer.read_text())["path"]).resolve()
    if not candidate.is_relative_to((root / "runs").resolve()):
        raise HTTPException(500, "Invalid experiment path")
    return candidate


@router.get("/experiment/status")
def experiment_status():
    if not (ROOT / "target_experiment/latest_run.json").is_file():
        return {"status": "running", "available": False}
    return {"available": True, **json.loads((experiment_directory() / "run_manifest.json").read_text())}


@router.get("/experiment/download/{filename}")
def download_experiment_result(filename: str):
    if filename not in ALLOWED | {"model_summary.csv"}:
        raise HTTPException(404, "Unknown experiment file")
    path = experiment_directory() / filename
    if not path.is_file():
        raise HTTPException(404, "Experiment result not available")
    return FileResponse(path, filename=filename)


@router.get("/portfolio/status")
def portfolio_status():
    path = ROOT / "portfolio_evaluation/portfolio_manifest.json"
    return {"available": True, **json.loads(path.read_text())} if path.is_file() else {"available": False}


@router.get("/portfolio/download/{filename}")
def download_portfolio_result(filename: str):
    if filename not in {"portfolio_holdings.csv", "portfolio_weekly_returns.csv", "portfolio_summary.csv",
                        "portfolio_nav_segments.csv", "portfolio_manifest.json"}:
        raise HTTPException(404, "Unknown portfolio file")
    path = ROOT / "portfolio_evaluation" / filename
    if not path.is_file():
        raise HTTPException(404, "Portfolio report not available")
    return FileResponse(path, filename=filename)
