import json
import os
import base64
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from web.services.weekly_updates import WeeklyUpdates, latest_period, PIPELINE_REVISION, DATA_FILES
from weekly_data_pipeline import build, generate_report, run_legacy_workflows


class WeeklyUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = datetime(2026, 11, 13, 3, tzinfo=timezone.utc)
        self.service = WeeklyUpdates(self.tmp.name, clock=lambda: self.now)
        self.env = patch.dict(os.environ, {"ANTHROPIC_API_KEY": "unit-test-key", "ANSEL_REPORT_OUTBOUND_ENABLED": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def write_legacy_outputs(self, directory, period):
        directory = Path(directory)
        figures = directory / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        period = pd.Timestamp(period)
        history = pd.DataFrame([{
            "date": period, "symbol": "BTC", "return": .01,
            "compound_weekly": .1, "extremely_positive_ratio": .1,
            "extremely_negative_ratio": .1,
        }])
        history.to_csv(directory / "df_merged_history.csv", index=False)
        prices = pd.DataFrame([{
            "date": period, "symbol": "BTC", "open": 100., "high": 102.,
            "low": 99., "close": 101., "usd_volume": 10000., "base_volume": 100.,
        }])
        prices.to_csv(directory / "stage_1_crypto_data.csv", index=False)
        news = pd.DataFrame([{
            "date": period, "id": "1", "title": "Bitcoin gains support",
            "body": "Bitcoin gains support despite fear", "positive": 1,
        }])
        scored = news.assign(reviewText=news.body, compound=.1, pos=.3, neg=.2, neu=.5,
                             sentiment=1)
        weekly = history.drop(columns=["symbol", "return"])
        for target in (directory, figures):
            news.to_csv(target / "stage_1_news_raw.csv", index=False)
            scored.drop(columns=["compound", "pos", "neg", "neu", "sentiment"]).to_csv(
                target / "stage_1_news_clean.csv", index=False)
            scored.to_csv(target / "stage_2_news_sentiment.csv", index=False)
            weekly.to_csv(target / "stage_2_sentiment_weekly.csv", index=False)
            pd.DataFrame({"word": ["support", "fear"], "count": [1, 1]}).to_csv(
                target / "stage_1_common_words.csv", index=False)
        for model in ("enet_all", "enet_market", "extra_all", "extra_market"):
            pd.DataFrame({"symbol": ["BTC"], "y_pred": [.02]}).to_csv(
                directory / f"pred_{model}.csv", index=False)
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aFOkAAAAASUVORK5CYII=")
        for name in ("wordcloud_positive", "wordcloud_negative", "fear_greed_gauge"):
            (figures / f"{name}.png").write_bytes(png)

    def publish(self, service=None, period="2026-11-11", report="complete", suffix="test",
                revision=PIPELINE_REVISION):
        service = service or self.service
        directory = service.root / "versions" / f"{period}_{suffix}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"period": period, "pipeline_revision": revision,
                    "report_status": report, "generated_at": self.now.isoformat()}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        self.write_legacy_outputs(directory, period)
        for name in DATA_FILES + (["figures/weekly_report.md"] if report == "complete" else []):
            path = directory / name
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_text("test publication")
        service.publish(period, directory, manifest)
        return directory

    def test_complete_period_utc_and_buffer_including_year_boundary(self):
        for now, expected in [("2026-11-11T23:59:00+00:00", "2026-11-04"),
                              ("2026-11-12T00:59:59+00:00", "2026-11-04"),
                              ("2026-11-12T01:00:00+00:00", "2026-11-11"),
                              ("2026-01-01T01:00:00+00:00", "2025-12-31"),
                              ("2026-11-12T09:00:00+08:00", "2026-11-11")]:
            self.assertEqual(latest_period(datetime.fromisoformat(now)), expected)

    def test_many_visitors_create_one_durable_job(self):
        services = [WeeklyUpdates(self.tmp.name, clock=lambda: self.now) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda i: services[i % 4].request(), range(24)))
        self.assertEqual(sum(r["status"] == "started" for r in results), 1)
        self.assertEqual(len({r["job_id"] for r in results}), 1)

    def test_two_workers_execute_one_job_and_keep_requests_responsive(self):
        self.service.request()
        other = WeeklyUpdates(self.tmp.name, clock=lambda: self.now)
        started, finish = threading.Event(), threading.Event()
        calls = []
        def runner(service, job):
            calls.append(job["id"])
            started.set()
            finish.wait(2)
            self.publish(service)
            return {"report_status": "complete"}
        thread = threading.Thread(target=lambda: self.service.run_pending(runner))
        thread.start()
        self.assertTrue(started.wait(2))
        self.assertFalse(other.run_pending(runner))
        self.assertEqual(other.request()["status"], "already_running")
        finish.set()
        thread.join(3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(other.request()["status"], "up_to_date")

    def test_failure_keeps_old_version_and_cooldown_survives_restart(self):
        old = self.publish(period="2026-11-04")
        self.service.request()
        def fail(*_):
            raise RuntimeError("provider unavailable")
        self.service.run_pending(fail)
        other = WeeklyUpdates(self.tmp.name, clock=lambda: self.now)
        self.assertEqual(other.published()["path"], old)
        self.assertEqual(other.request()["status"], "cooldown")
        self.now += timedelta(minutes=16)
        self.assertEqual(other.request()["status"], "started")

    def test_report_is_not_fresh_merely_because_data_is_fresh(self):
        self.publish(report="failed")
        self.assertEqual(self.service.request(component="data")["status"], "up_to_date")
        self.assertEqual(self.service.request(component="report")["status"], "started")

    def test_missing_ai_configuration_does_not_repeat_data_work(self):
        self.publish(report="not_configured")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            self.assertEqual(self.service.request(component="all")["status"], "up_to_date")
            self.assertFalse(self.service.status("report")["can_update"])

    def test_interrupted_worker_recovered_without_marking_partial_files_current(self):
        job = self.service.request()
        with self.service.connect() as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (job["job_id"],))
        self.assertFalse(self.service.run_pending())
        self.assertFalse(self.service.status()["data_current"])
        self.assertEqual(self.service.status()["last_job"]["status"], "failed")

    def test_completed_orphan_worker_result_is_recovered(self):
        job = self.service.request()
        self.publish()
        (self.service.root / f"{job['job_id']}.result.json").write_text('{"report_status":"complete"}')
        with self.service.connect() as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (job["job_id"],))
        self.service.run_pending()
        self.assertEqual(self.service.status()["last_job"]["status"], "succeeded")

    def test_new_calendar_week_triggers_catchup(self):
        self.publish()
        self.now = datetime(2026, 11, 19, 1, tzinfo=timezone.utc)
        result = self.service.request(trigger="scheduled")
        self.assertEqual(result["status"], "started")
        self.assertEqual(result["target_period"], "2026-11-18")

    def test_api_aliases_cannot_force_regenerate_a_current_report(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from web.routers import api_updates, api_analysis, api_sentiment
        self.publish()
        app = FastAPI()
        for router in [api_updates.router, api_analysis.router, api_sentiment.router]:
            app.include_router(router, prefix="/api")
        with patch.object(api_updates, "get_updates", return_value=self.service), \
             patch.object(api_analysis, "get_updates", return_value=self.service), \
             patch.object(api_sentiment, "get_updates", return_value=self.service):
            client = TestClient(app)
            self.assertEqual(client.post('/api/analysis/generate', json={"force": True}).json()["status"], "up_to_date")
            self.assertEqual(client.post('/api/sentiment/update').json()["status"], "up_to_date")
            self.assertEqual(client.post('/api/updates/run', json={"component": "all"}).json()["status"], "up_to_date")
            self.assertEqual(client.get('/api/updates/status?component=unknown').status_code, 400)

    def test_no_placeholder_report_when_ai_is_unconfigured(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            status, _ = generate_report(Path(self.tmp.name), pd.Timestamp("2026-11-11"))
        self.assertEqual(status, "not_configured")
        self.assertFalse((Path(self.tmp.name) / "figures/weekly_report.md").exists())

    def test_unapproved_outbound_report_is_blocked_before_loading_client(self):
        self.publish(report="failed")
        with patch.dict(os.environ, {"ANSEL_REPORT_OUTBOUND_ENABLED": "0"}), \
             patch('weekly_data_pipeline.importlib.util.spec_from_file_location', side_effect=AssertionError('must not load API client')):
            status, _ = generate_report(Path(self.tmp.name), pd.Timestamp('2026-11-11'))
            self.assertEqual(status, 'approval_required')
            self.assertFalse(self.service.status('report')['can_update'])
            self.assertEqual(self.service.request(component='all')['status'], 'up_to_date')

    def test_missing_artifact_prevents_false_latest_status(self):
        path = self.publish()
        (path / 'pred_extra_all.csv').unlink()
        self.assertFalse(self.service.status('data')['data_current'])

    def test_report_retry_reuses_data_without_download_or_training(self):
        self.publish(report="failed")
        def report(directory, period):
            (directory / 'figures/weekly_report.md').write_text('new report')
            return 'complete', None
        with patch('weekly_data_pipeline.run_legacy_workflows', side_effect=AssertionError('data should be reused')), \
             patch('weekly_data_pipeline.generate_report', side_effect=report):
            build(self.service, 'retry-job', '2026-11-11')
        self.assertTrue(self.service.status('report')['report_current'])

    def test_workflow_wrapper_uses_legacy_entries_with_isolated_output(self):
        prediction = Mock()
        sentiment = Mock()
        output = Path(self.tmp.name) / "isolated"
        history = Path(self.tmp.name) / "history.csv"
        history.write_text("date,symbol,return,audit_metadata\n2026-11-04,BTC,0.01,99\n")
        original_history = history.read_bytes()
        root = Path(self.tmp.name) / "original-project"
        root.mkdir()
        (root / "df_merged_history.csv").write_text("date,symbol,return\n")
        period = pd.Timestamp("2026-11-11")
        modules = {
            "customer_side": SimpleNamespace(run_for_client=prediction),
            "sentiment_update": SimpleNamespace(update_sentiment_and_gauge=sentiment),
        }
        price_loader, news_loader = Mock(), Mock()
        with patch.dict(sys.modules, modules), \
             patch('weekly_data_pipeline.ROOT', root), \
             patch('web.services.legacy_data_sources.make_legacy_loaders',
                   return_value=(price_loader, news_loader)):
            run_legacy_workflows(output, period, history, service=self.service)
        prediction.assert_called_once()
        sentiment.assert_called_once()
        self.assertEqual(Path(prediction.call_args.args[1]), output / "df_merged_history.csv")
        self.assertEqual(history.read_bytes(), original_history)
        self.assertEqual(pd.read_csv(output / "df_merged_history.csv").columns.tolist(),
                         ["date", "symbol", "return"])
        self.assertEqual(prediction.call_args.kwargs["output_dir"], output)
        self.assertEqual(prediction.call_args.kwargs["decision_date"], period)
        self.assertTrue(prediction.call_args.kwargs["refresh"])
        self.assertIs(prediction.call_args.kwargs["price_loader"], price_loader)
        self.assertIs(prediction.call_args.kwargs["news_loader"], news_loader)
        self.assertEqual(sentiment.call_args.kwargs["output_dir"], output)
        self.assertEqual(sentiment.call_args.kwargs["as_of"], period)
        self.assertIs(sentiment.call_args.kwargs["news_loader"], news_loader)

    def test_revision_upgrade_rebuilds_same_week_without_changing_old_version(self):
        old = self.publish(revision=PIPELINE_REVISION - 1)
        old_manifest = (old / "manifest.json").read_bytes()
        old_history = (old / "df_merged_history.csv").read_bytes()
        self.assertFalse(self.service.status("data")["data_current"])
        self.assertEqual(self.service.request(component="data")["status"], "started")
        def workflows(directory, period, history_path, **kwargs):
            self.assertEqual(Path(history_path), old / "df_merged_history.csv")
            self.write_legacy_outputs(directory, period)
        with patch('weekly_data_pipeline.run_legacy_workflows', side_effect=workflows) as legacy, \
             patch('weekly_data_pipeline.generate_report', return_value=('approval_required', 'disabled')):
            build(self.service, 'revision-upgrade', '2026-11-11')
        legacy.assert_called_once()
        self.assertNotEqual(self.service.published()["path"], old)
        self.assertTrue(self.service.status("data")["data_current"])
        self.assertEqual((old / "manifest.json").read_bytes(), old_manifest)
        self.assertEqual((old / "df_merged_history.csv").read_bytes(), old_history)

    def test_legacy_workflow_failure_cannot_publish_partial_output(self):
        old = self.publish(period="2026-11-04")
        def fail(directory, period, history_path, **kwargs):
            figures = directory / "figures"
            figures.mkdir(parents=True)
            (figures / "wordcloud_positive.png").write_bytes(b"unfinished")
            raise RuntimeError("legacy data source unavailable")
        with patch('weekly_data_pipeline.run_legacy_workflows', side_effect=fail), \
             patch('weekly_data_pipeline.generate_report') as report:
            with self.assertRaisesRegex(RuntimeError, "legacy data source unavailable"):
                build(self.service, 'failed-legacy-job', '2026-11-11')
        self.assertEqual(self.service.published()["path"], old)
        report.assert_not_called()

    def test_invalid_predictions_do_not_replace_published_data(self):
        old = self.publish(period="2026-11-04")
        def workflows(directory, period, history_path, **kwargs):
            self.write_legacy_outputs(directory, period)
            pd.DataFrame({"symbol": ["BTC"], "y_pred": [float("inf")]}).to_csv(
                directory / "pred_extra_all.csv", index=False)
        with patch('weekly_data_pipeline.run_legacy_workflows', side_effect=workflows), \
             patch('weekly_data_pipeline.generate_report') as report:
            with self.assertRaises(ValueError):
                build(self.service, 'invalid-predictions', '2026-11-11')
        self.assertEqual(self.service.published()["path"], old)
        report.assert_not_called()


if __name__ == '__main__':
    unittest.main()
