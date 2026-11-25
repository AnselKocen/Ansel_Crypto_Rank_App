"""
Claude AI Service
Wraps existing claude_assistant.py functionality for chat
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
import json

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from claude_assistant import (
    CryptoAssistant,
    get_preset_questions,
    handle_preset_question,
    FIXED_PROJECT_INTRO
)


def get_presets() -> List[Dict]:
    """Get list of preset questions"""
    return get_preset_questions()


def check_preset_response(question_id: str) -> Optional[str]:
    """
    Check if preset question has fixed response
    Returns the fixed response or None if should use AI
    """
    return handle_preset_question(question_id)


def get_project_intro() -> str:
    """Get the fixed project introduction"""
    return FIXED_PROJECT_INTRO


class ChatSession:
    """Manages a chat session with Claude"""

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.assistant = CryptoAssistant(api_key, model)
        self.messages: List[Dict] = []

    def is_available(self) -> bool:
        return self.assistant.is_available()

    def send_message(self, message: str, context: str = "") -> str:
        """Send a message and get response"""
        if not self.is_available():
            return "Error: Claude API key not configured"

        response = self.assistant.chat(message, context)
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        """Clear chat history"""
        self.assistant.clear_history()
        self.messages = []

    def get_history(self) -> List[Dict]:
        """Get chat history"""
        return self.messages


# Session storage (in-memory, per request)
_sessions: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: str, api_key: str) -> ChatSession:
    """Get existing session or create new one"""
    if session_id not in _sessions or _sessions[session_id].assistant.api_key != api_key:
        _sessions[session_id] = ChatSession(api_key)
    return _sessions[session_id]


def clear_session(session_id: str):
    """Clear a session"""
    if session_id in _sessions:
        del _sessions[session_id]
