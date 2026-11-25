"""
Skills/Knowledge Base API Routes
"""

from fastapi import APIRouter, HTTPException
from typing import Literal
from web.services.skills_service import (
    get_skills_list,
    get_skill_content,
    get_categorized_skills
)

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("")
async def get_skills():
    """Get all skills with categories"""
    skills = get_skills_list()
    categories = get_categorized_skills()
    return {
        "skills": skills,
        "categories": categories
    }


@router.get("/{name}")
async def get_skill(name: str, lang: Literal["zh", "en"] = "zh"):
    """Get skill content by name"""
    content = get_skill_content(name, lang)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return {
        "name": name,
        "lang": lang,
        "content": content
    }
