"""
AI Market Analysis Service
Wraps existing ai_market_analysis.py functionality
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai_market_analysis import (
    should_update_report,
    generate_analysis_report as _generate_report,
    load_existing_report,
    get_report_last_update,
    export_report_to_file
)


def get_report_status() -> dict:
    """Get current status of the analysis report"""
    should_update, reason = should_update_report()
    last_update = get_report_last_update()
    report_exists = load_existing_report() is not None

    return {
        "should_update": should_update,
        "reason": reason,
        "last_update": last_update,
        "report_exists": report_exists
    }


def get_report_content() -> Optional[str]:
    """Get existing report content if available"""
    from web.services.weekly_updates import data_path
    path = data_path("figures/weekly_report.md")
    return path.read_text(encoding="utf-8") if path.is_file() else None


async def generate_report(api_key: str, force: bool = False) -> dict:
    """
    Generate a new analysis report

    Returns dict with:
        - content: Report content
        - was_generated: Whether new report was generated
        - message: Status message
    """
    try:
        content, was_generated, message = _generate_report(api_key, force)
        return {
            "success": True,
            "content": content,
            "was_generated": was_generated,
            "message": message
        }
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "was_generated": False,
            "message": f"Error: {str(e)}"
        }


def export_report(content: str, filename: str = "crypto_report.md") -> str:
    """Export report to file and return path"""
    return export_report_to_file(content, filename)
