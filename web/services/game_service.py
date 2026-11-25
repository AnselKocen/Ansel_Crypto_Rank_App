"""
Game Service
Loads question and cat data for the Crypto Cat Adventure game.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
QUESTIONS_DIR = BASE_DIR / "game" / "questions"
LEVELS_DIR = BASE_DIR / "game" / "levels"
CATS_FILE = BASE_DIR / "game" / "cats" / "cat_data.json"


def get_all_levels() -> List[Dict]:
    """Get metadata for all levels (without questions)."""
    levels = []
    for i in range(1, 11):
        fpath = QUESTIONS_DIR / f"level_{i:02d}.json"
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            levels.append({
                "level": data["level"],
                "world_name_zh": data["world_name_zh"],
                "theme_zh": data["theme_zh"],
                "difficulty": data["difficulty"],
                "story_zh": data["story_zh"],
                "question_count": len(data.get("questions", [])),
            })
        except Exception as e:
            print(f"[GameService] Error loading level {i}: {e}")
    return levels


def get_level_data(level_id: int) -> Optional[Dict]:
    """Get full level data including questions."""
    fpath = QUESTIONS_DIR / f"level_{level_id:02d}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[GameService] Error loading level {level_id}: {e}")
        return None


def get_cats_data() -> List[Dict]:
    """Get all cat character data."""
    if not CATS_FILE.exists():
        return []
    try:
        with open(CATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cats", [])
    except Exception as e:
        print(f"[GameService] Error loading cats: {e}")
        return []


def get_level_map(level_id: int) -> Optional[Dict]:
    """Get level map data (platforms, coins, question blocks) for the platformer."""
    fpath = LEVELS_DIR / f"level_{level_id:02d}_map.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[GameService] Error loading level map {level_id}: {e}")
        return None
