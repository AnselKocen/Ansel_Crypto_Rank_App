"""
Game API Router
Serves question and cat data for the Crypto Cat Adventure game.
"""

from fastapi import APIRouter, HTTPException
from web.services.game_service import get_all_levels, get_level_data, get_cats_data, get_level_map

router = APIRouter(prefix="/game", tags=["game"])


@router.get("/levels")
async def list_levels():
    """Get all level metadata (without questions)."""
    levels = get_all_levels()
    return {"levels": levels}


@router.get("/levels/{level_id}")
async def get_level(level_id: int):
    """Get full level data with questions."""
    if level_id < 1 or level_id > 10:
        raise HTTPException(status_code=404, detail=f"Level {level_id} not found")
    data = get_level_data(level_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Level {level_id} not found")
    return data


@router.get("/levels/{level_id}/map")
async def get_level_map_data(level_id: int):
    """Get level map data (platforms, coins, blocks) for the platformer."""
    if level_id < 1 or level_id > 10:
        raise HTTPException(status_code=404, detail=f"Level {level_id} map not found")
    data = get_level_map(level_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Level {level_id} map not found")
    return data


@router.get("/cats")
async def list_cats():
    """Get all cat character data."""
    cats = get_cats_data()
    return {"cats": cats}
