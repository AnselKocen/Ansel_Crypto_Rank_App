"""
AI Chat API Routes
"""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Literal, Optional
from web.services.kimi_service import (
    ChatServiceError,
    ChatSettings,
    get_presets,
    get_settings,
    resolve_question,
    get_or_create_session,
    clear_session
)
from web.services.chat_presets import get_preset

router = APIRouter(prefix="/chat", tags=["chat"])


def _error_response(code: str, language: str):
    """Return safe text for direct API clients as well as the bilingual UI."""
    messages = {
        "not_configured": ("AI 问答尚未配置 API Key。", "The AI API key has not been configured."),
        "provider_timeout": ("AI 回答超时，请稍后重试。", "The AI request timed out. Please try again."),
        "invalid_api_key": ("AI 密钥验证失败，请检查服务端配置。", "The AI API key could not be authenticated."),
        "rate_limited": ("AI 服务请求较多，请稍后重试。", "The AI service is rate limited. Please try again later."),
        "knowledge_unavailable": ("暂时无法核实所需资料，请重试或换一种问法。", "The required sources could not be verified. Please try again."),
    }
    fallback = ("AI 服务暂时不可用，请稍后重试。", "The AI service is temporarily unavailable. Please try again.")
    return {"response": messages.get(code, fallback)[1 if language == "en" else 0],
            "error": True, "error_code": code}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    preset_id: Optional[Literal["intro", "interpret", "risk", "models", "crypto_basics", "get_crypto"]] = None
    lang: Literal["zh", "en"] = "zh"
    selected_document_id: Optional[str] = Field(default=None, max_length=160)


class ClearRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)


@router.get("/presets")
async def get_preset_questions():
    """Get preset questions for UI"""
    presets = get_presets()
    return {"presets": presets}


@router.post("/message")
async def send_message(request: ChatRequest):
    """Quick answers stay local; free questions use the Kimi knowledge loop."""
    try:
        preset = get_preset(request.preset_id) if request.preset_id else None
        try:
            settings = get_settings()
        except ChatServiceError:
            if not preset:
                raise
            settings = ChatSettings()
        if not settings.api_key and not preset:
            raise ChatServiceError("not_configured")
        session = await run_in_threadpool(
            get_or_create_session,
            request.session_id, settings.api_key, model=settings.model,
            base_url=settings.base_url, reasoning_effort=settings.reasoning_effort,
        )
        question = resolve_question(request.message, request.preset_id, request.lang)
        if preset:
            response = preset["response_en" if request.lang == "en" else "response_zh"]
            await run_in_threadpool(session.record_fixed_response, question, response)
            return {"response": response, "is_preset": True, "provider": "local", "preset_id": preset["id"]}
        response = await run_in_threadpool(
            session.send_message, question, language=request.lang,
            selected_document_id=request.selected_document_id,
            require_project=request.preset_id in {"intro", "interpret", "models"},
        )
        return {"response": response, "is_preset": False, "provider": "kimi"}
    except ChatServiceError as error:
        return _error_response(error.code, request.lang)
    except Exception:
        # Provider exceptions can contain authorization headers or raw input.
        return _error_response("provider_unavailable", request.lang)


@router.post("/clear")
async def clear_chat(request: ClearRequest):
    """Clear chat history for a session"""
    await run_in_threadpool(clear_session, request.session_id)
    return {"status": "cleared"}


@router.get("/project-intro")
async def get_intro():
    """Compatibility endpoint: serve the same current public source used by Kimi."""
    from web.services.knowledge_service import execute_tool
    document = execute_tool("read_knowledge", {"id": "project/overview"})
    return {"content": document.get("content", ""), "source": document.get("id")}
