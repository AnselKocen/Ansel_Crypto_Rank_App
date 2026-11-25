"""
Figures/Images API Routes
"""

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/figures", tags=["figures"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = BASE_DIR / "figures"


@router.get("/list")
async def list_figures():
    """List all available figures"""
    if FIGURES_DIR.exists():
        files = [f.name for f in FIGURES_DIR.iterdir() if f.is_file()]
        return {"figures": files}
    return {"figures": []}


@router.get("/{filename:path}")
async def get_figure(filename: str):
    """Get a specific figure file"""
    file_path = FIGURES_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}
