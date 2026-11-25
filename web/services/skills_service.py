"""
Skills/Knowledge Base Service
Wraps existing claude_assistant.py skills functionality
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from claude_assistant import (
    get_skills_list as _get_skills_list,
    get_skill_content as _get_skill_content,
    categorize_skills as _categorize_skills
)
from web.services.knowledge_service import SKILLS_ZH

TRANSLATIONS_DIR = BASE_DIR / "web/content/skills/zh"


def get_skills_list() -> List[Dict[str, str]]:
    """Get list of all available skills"""
    return _get_skills_list()


def get_skill_content(skill_name: str, language: str = "en") -> Optional[str]:
    """Get content for a specific skill"""
    if skill_name not in SKILLS_ZH:
        return None
    if language == "en":
        return _get_skill_content(skill_name)
    if language != "zh":
        return None
    source_dir = BASE_DIR / "skills" / skill_name
    translated_dir = TRANSLATIONS_DIR / skill_name
    paths = [translated_dir / "SKILL.md"]
    paths.extend(translated_dir / "references" / source.name
                 for source in sorted((source_dir / "references").glob("*.md")))
    # IDs come from the public catalogue; translations never accept a file path.
    if any(not path.resolve().is_relative_to(TRANSLATIONS_DIR.resolve()) or not path.is_file()
           for path in paths):
        return None
    parts = []
    for index, path in enumerate(paths):
        text = path.read_text(encoding="utf-8")
        parts.append(text if index == 0 else f"\n\n### 参考资料：{path.stem}\n\n{text}")
    return "".join(parts)


def get_categorized_skills() -> Dict[str, List[Dict]]:
    """Get skills organized by category"""
    skills = _get_skills_list()
    return _categorize_skills(skills)
