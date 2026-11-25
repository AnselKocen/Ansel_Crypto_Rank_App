"""Static Chinese documents and their language-aware public endpoint."""
from pathlib import Path
import re
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routers import api_skills
from web.services.skills_service import BASE_DIR, TRANSLATIONS_DIR, get_skill_content


class SkillLocalizationTests(unittest.TestCase):
    def test_every_source_document_has_a_chinese_body_with_preserved_links(self):
        sources = sorted((BASE_DIR / "skills").glob("*/SKILL.md"))
        sources += sorted((BASE_DIR / "skills").glob("*/references/*.md"))
        self.assertEqual(len(sources), 48)
        for source in sources:
            with self.subTest(document=str(source)):
                translated = TRANSLATIONS_DIR / source.relative_to(BASE_DIR / "skills")
                self.assertTrue(translated.is_file())
                original, chinese = source.read_text(), translated.read_text()
                self.assertRegex(chinese, r"[\u4e00-\u9fff]{3,}")
                self.assertNotIn("When to Use", chinese)
                links = re.findall(r"https?://[^\s)<>]+", original)
                self.assertTrue(all(link in chinese for link in links))

    def test_language_endpoint_and_selected_reference_content(self):
        app = FastAPI()
        app.include_router(api_skills.router, prefix="/api")
        with TestClient(app) as client:
            slug = "cryptocurrency-investment-strategy-and-due-diligence"
            zh = client.get(f"/api/skills/{slug}")
            en = client.get(f"/api/skills/{slug}?lang=en")
            self.assertEqual(zh.status_code, 200)
            self.assertEqual(en.status_code, 200)
            self.assertEqual(zh.json()["lang"], "zh")
            self.assertIn("When to Use", en.json()["content"])
            self.assertNotIn("When to Use", zh.json()["content"])
            self.assertRegex(zh.json()["content"], r"[\u4e00-\u9fff]")
            self.assertEqual(client.get(f"/api/skills/{slug}?lang=fr").status_code, 422)
            self.assertEqual(client.get("/api/skills/not-a-document").status_code, 404)
        content = get_skill_content("bitcoin-market-cycle-analysis", "zh")
        self.assertIn("参考资料", content)

    def test_unknown_name_cannot_select_a_file(self):
        for name in ("../.env", ".env", "bitcoin-market-cycle-analysis/../../.env"):
            self.assertIsNone(get_skill_content(name, "zh"))


if __name__ == "__main__":
    unittest.main()
