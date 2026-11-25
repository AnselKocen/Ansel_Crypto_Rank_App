"""Chat API contract checks with a stubbed Kimi session and settings."""
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routers import api_chat
from web.services import kimi_service


class KimiChatApiTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api_chat.router, prefix="/api")
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.settings = kimi_service.ChatSettings(
            api_key="offline-router-secret", model="kimi-k3",
            base_url="https://api.moonshot.cn/v1", reasoning_effort="low")
        self.session = Mock()
        self.session.is_available.return_value = True
        self.session.send_message.return_value = "A stubbed Kimi response."
        settings_patch = patch.object(api_chat, "get_settings", return_value=self.settings)
        session_patch = patch.object(api_chat, "get_or_create_session", return_value=self.session)
        settings_patch.start()
        self.get_session = session_patch.start()
        self.addCleanup(settings_patch.stop)
        self.addCleanup(session_patch.stop)

    def test_custom_message_forwards_language_document_and_session(self):
        response = self.client.post("/api/chat/message", json={
            "message": "Explain the selected document", "session_id": "session-a", "lang": "en",
            "selected_document_id": "skill/bitcoin-transaction-model",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "A stubbed Kimi response.")
        call = self.session.send_message.call_args
        self.assertEqual(call.args[0], "Explain the selected document")
        self.assertEqual(call.kwargs["language"], "en")
        self.assertEqual(call.kwargs["selected_document_id"], "skill/bitcoin-transaction-model")
        self.assertFalse(call.kwargs.get("require_project", False))
        self.assertEqual(self.get_session.call_args.args[:2], ("session-a", "offline-router-secret"))
        self.assertNotIn("offline-router-secret", response.text)

    def test_project_intro_is_local_and_records_followup_context(self):
        response = self.client.post("/api/chat/message", json={
            "message": "介绍这个项目", "preset_id": "intro", "session_id": "intro-session", "lang": "zh",
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_preset"])
        self.assertEqual(response.json()["provider"], "local")
        self.assertIn("👋", response.json()["response"])
        self.session.send_message.assert_not_called()
        self.session.record_fixed_response.assert_called_once()
        self.assertEqual(self.session.record_fixed_response.call_args.args[1], response.json()["response"])

    def test_all_presets_work_in_both_languages_without_an_api_key(self):
        from web.services.chat_presets import fixed_answer, get_presets
        presets = get_presets()
        self.assertTrue({"crypto_basics", "get_crypto"}.issubset({preset["id"] for preset in presets}))
        for preset in presets:
            for language in ("zh", "en"):
                with self.subTest(preset=preset["id"], language=language), patch.object(
                    api_chat, "get_settings", return_value=kimi_service.ChatSettings()
                ):
                    self.session.record_fixed_response.reset_mock()
                    response = self.client.post("/api/chat/message", json={
                        "message": "Untrusted browser text", "preset_id": preset["id"],
                        "session_id": "preset-session", "lang": language,
                    })
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["response"], fixed_answer(preset["id"], language))
                    self.assertEqual(data["provider"], "local")
                    self.assertTrue(data["is_preset"])
                    self.assertEqual(data["preset_id"], preset["id"])
                    self.session.record_fixed_response.assert_called_once()
                    self.assertEqual(self.session.record_fixed_response.call_args.args,
                                     (preset["label_en" if language == "en" else "label"], data["response"]))
                    self.assertNotIn("offline-router-secret", response.text)
        self.session.send_message.assert_not_called()

    def test_unknown_preset_id_is_rejected_before_creating_a_session(self):
        response = self.client.post("/api/chat/message", json={
            "message": "An unknown shortcut", "preset_id": "unknown-preset", "lang": "zh",
        })
        self.assertEqual(response.status_code, 422)
        self.get_session.assert_not_called()
        self.session.record_fixed_response.assert_not_called()
        self.session.send_message.assert_not_called()

    def test_preset_is_independent_of_broken_model_settings(self):
        with patch.object(api_chat, "get_settings", side_effect=kimi_service.ChatServiceError("invalid_configuration")):
            response = self.client.post("/api/chat/message", json={
                "message": "介绍这个项目", "preset_id": "intro", "session_id": "preset-session"})
        self.assertFalse(response.json().get("error", False))
        self.session.send_message.assert_not_called()

    def test_unconfigured_key_never_creates_a_provider_session(self):
        settings = kimi_service.ChatSettings(
            api_key="", model="kimi-k3", base_url="https://api.moonshot.cn/v1", reasoning_effort="low")
        with patch.object(api_chat, "get_settings", return_value=settings):
            response = self.client.post("/api/chat/message", json={"message": "Hello"})
        self.assertTrue(response.json()["error"])
        self.assertEqual(response.json()["error_code"], "not_configured")
        self.get_session.assert_not_called()
        self.session.send_message.assert_not_called()

    def test_provider_failure_returns_safe_localized_error(self):
        self.session.send_message.side_effect = kimi_service.ChatServiceError("provider_timeout")
        response = self.client.post("/api/chat/message", json={"message": "你好", "lang": "zh"})
        data = response.json()
        self.assertTrue(data["error"])
        self.assertEqual(data["error_code"], "provider_timeout")
        self.assertNotIn("offline-router-secret", response.text)
        self.assertRegex(data["response"], r"[\u4e00-\u9fff]")

    def test_unexpected_failure_does_not_expose_exception_text_or_key(self):
        self.session.send_message.side_effect = RuntimeError("private failure offline-router-secret")
        response = self.client.post("/api/chat/message", json={"message": "Hello", "lang": "en"})
        self.assertTrue(response.json()["error"])
        self.assertNotIn("offline-router-secret", response.text)
        self.assertNotIn("private failure", response.text)

    def test_clear_targets_only_the_requested_session(self):
        with patch.object(api_chat, "clear_session") as clear:
            response = self.client.post("/api/chat/clear", json={"session_id": "one-session"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cleared")
        clear.assert_called_once_with("one-session")
        self.session.send_message.assert_not_called()

    def test_loading_presets_never_calls_the_model(self):
        response = self.client.get("/api/chat/presets")
        self.assertEqual(response.status_code, 200)
        presets = response.json()["presets"]
        self.assertEqual(len(presets), 6)
        self.assertEqual({preset["id"] for preset in presets},
                         {"intro", "interpret", "risk", "models", "crypto_basics", "get_crypto"})
        for preset in presets:
            self.assertEqual(preset["type"], "fixed")
            self.assertTrue(preset["label"])
            self.assertTrue(preset["label_en"])
        self.get_session.assert_not_called()
        self.session.send_message.assert_not_called()
        self.assertNotIn("offline-router-secret", response.text)


if __name__ == "__main__":
    unittest.main()
