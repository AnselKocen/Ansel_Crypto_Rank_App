"""Contract, pagination, and allowlist checks for the local Q&A tools."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from web.services import knowledge_service as knowledge


class KnowledgeServiceTests(unittest.TestCase):
    def test_registry_validation_performs_no_reads(self):
        with patch.object(knowledge, "_public_text", side_effect=AssertionError("read")), \
             patch.object(knowledge, "_snapshot", side_effect=AssertionError("state read")):
            self.assertTrue(knowledge.is_known_document("skill/bitcoin-market-cycle-analysis"))
            self.assertTrue(knowledge.is_known_document("project/status"))
            self.assertTrue(knowledge.is_known_document("overview"))
            self.assertFalse(knowledge.is_known_document("../../.env"))
            self.assertFalse(knowledge.is_known_document(["project/status"]))

    def test_real_corpus_and_bilingual_search(self):
        docs = knowledge._catalogue()
        self.assertEqual(sum(d["id"].startswith("skill/") for d in docs), 36)
        self.assertEqual(sum(d["id"].startswith("reference/") for d in docs), 12)
        for query, expected in [
            ("冷钱包备份", "skill/cryptocurrency-wallet-management"),
            ("模型训练特征", "project/models"),
            ("latest data status", "project/status"),
        ]:
            result = knowledge.search_knowledge(query)
            self.assertIn(expected, [d["id"] for d in result["results"]])
            self.assertLessEqual(len(result["results"]), 5)
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 3500)
        result = knowledge.search_knowledge("比特币减半", scope="references")
        self.assertTrue(result["results"])
        self.assertTrue(all(d["id"].startswith("reference/") for d in result["results"]))

    def test_project_documents_have_canonical_ids_and_complete_content(self):
        for name in ("overview", "models", "updates", "status"):
            result = knowledge.read_knowledge(name)
            self.assertEqual(result["id"], "project/"+name)
            self.assertTrue(result["content"])
            self.assertFalse(result["truncated"])
            self.assertIsNone(result["continuation"])
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 6000)
        status = knowledge.read_knowledge("project/status")["content"]
        self.assertIn("checked_at_utc", status)
        self.assertIn("target_period", status)

    def test_single_read_and_scoped_search_do_not_read_other_sources(self):
        with patch.object(knowledge, "_public_text", wraps=knowledge._public_text) as reader, \
             patch.object(knowledge, "_project_document", side_effect=AssertionError("project read")):
            result = knowledge.read_knowledge("skill/bitcoin-economic-incentives")
            self.assertTrue(result["content"])
            reader.assert_called_once_with("skills/bitcoin-economic-incentives/SKILL.md")
            reader.reset_mock()
            result = knowledge.search_knowledge("wallet", scope="skills")
            self.assertTrue(result["results"])
            self.assertEqual(reader.call_count, 36)
            self.assertTrue(all(call.args[0].endswith("/SKILL.md") for call in reader.call_args_list))

    def test_path_requests_and_unlisted_tools_cannot_read(self):
        with patch.object(knowledge, "_catalogue", side_effect=AssertionError("unexpected read")):
            for identifier in [".env", "skill/../../.env", "/etc/passwd", "file:///etc/passwd"]:
                self.assertEqual(knowledge.read_knowledge(identifier)["error"], "unknown_document")
        with self.assertRaises(ValueError):
            knowledge._public_text(".env")
        self.assertEqual(knowledge.execute_tool("exec", {"cmd": "ls"})["error"], "unknown_tool")
        self.assertEqual(knowledge.execute_tool("read_knowledge", {"id": "project/status", "path": ".env"})["error"], "invalid_arguments")

    def test_symlinked_allowlisted_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            allowed = root / "skills/bitcoin-economic-incentives/SKILL.md"
            allowed.parent.mkdir(parents=True)
            hidden = root / ".env"
            hidden.write_text("private fixture")
            allowed.symlink_to(hidden)
            with patch.object(knowledge, "PROJECT_ROOT", root):
                with self.assertRaises(ValueError):
                    knowledge._public_text("skills/bitcoin-economic-incentives/SKILL.md")

    def test_long_lines_paginate_without_losing_content(self):
        original = "# Long section\n" + "连续正文"*3000 + "\nlast line\n"
        doc = knowledge._document("project/overview", "Example", "示例", "", "test", original)
        arguments = {"id": "project/overview", "max_lines": 2}
        pieces = []
        with patch.object(knowledge, "_get_document", return_value=doc):
            for _ in range(20):
                result = knowledge.read_knowledge(**arguments)
                self.assertLessEqual(len(result["content"]), knowledge.MAX_CONTENT_CHARS)
                self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 6000)
                pieces.append(result["content"])
                if result["continuation"] is None:
                    break
                arguments = result["continuation"]
            else:
                self.fail("Pagination did not finish")
        self.assertEqual("".join(pieces), original)

    def test_section_read_excludes_sibling_sections(self):
        doc = knowledge._document("project/overview", "Example", "示例", "", "test",
            "# Intro\nintro\n## Wallet\nwallet\n### Backup\nbackup\n## Other\nother\n")
        with patch.object(knowledge, "_get_document", return_value=doc):
            result = knowledge.read_knowledge("project/overview", section="Wallet")
        self.assertEqual(result["start_line"], 3)
        self.assertIn("backup", result["content"])
        self.assertNotIn("other", result["content"])
        self.assertFalse(result["truncated"])

    def test_publication_projection_excludes_private_manifest_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = {"pipeline_revision": 2, "visual_revision": 1,
                "generated_at": "2026-11-24T12:00:00+00:00", "report_status": "failed",
                "report_error": "PRIVATE ERROR DETAIL", "seed_history": "/private/path",
                "coverage": {"news_week_articles": 191, "note": "PRIVATE NOTE"}}
            with sqlite3.connect(root / "state.sqlite3") as db:
                db.executescript("CREATE TABLE publication(singleton INTEGER, period TEXT, path TEXT, manifest TEXT); CREATE TABLE jobs(status TEXT);")
                db.execute("INSERT INTO publication VALUES(1,?,?,?)", ("2026-11-18", "versions/public", json.dumps(manifest)))
            before = (root / "state.sqlite3").read_bytes()
            connections = []
            connect = sqlite3.connect
            def track_connection(*args, **kwargs):
                connection = connect(*args, **kwargs)
                connections.append(connection)
                return connection
            with patch.dict("os.environ", {"ANSEL_UPDATE_DIR": str(root)}), \
                 patch.object(knowledge.sqlite3, "connect", side_effect=track_connection):
                result = knowledge._snapshot()
            self.assertEqual(result["period"], "2026-11-18")
            self.assertEqual(result["coverage"], {"news_week_articles": 191})
            self.assertNotIn("PRIVATE", json.dumps(result))
            self.assertNotIn("/private/path", json.dumps(result))
            self.assertEqual(before, (root / "state.sqlite3").read_bytes())
            with self.assertRaises(sqlite3.ProgrammingError):
                connections[0].execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
