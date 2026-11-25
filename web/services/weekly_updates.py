"""Shared, durable weekly jobs and immutable published data versions.

One SQLite database and filesystem lock must be shared by all Web workers.
Heavy work runs in a separate Python process, never in an async request handler.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
PIPELINE_REVISION = 2
DATA_FILES = ["df_merged_history.csv", "stage_1_crypto_data.csv",
              *[f"{prefix}{name}.csv" for prefix in ("", "figures/") for name in
                ("stage_1_news_raw", "stage_1_news_clean", "stage_2_news_sentiment",
                 "stage_2_sentiment_weekly", "stage_1_common_words")], *[f"pred_{m}.csv" for m in
              ("enet_all", "enet_market", "extra_all", "extra_market")],
              *[f"figures/{name}.png" for name in
                ("wordcloud_positive", "wordcloud_negative", "fear_greed_gauge")]]


def utcnow():
    return datetime.now(UTC)


def latest_period(now=None):
    now = now or utcnow()
    if now.tzinfo is None:
        raise ValueError("A timezone-aware clock is required")
    complete_day = (now.astimezone(UTC)-timedelta(hours=1)).date()-timedelta(days=1)
    return (complete_day-timedelta(days=(complete_day.weekday()-2) % 7)).isoformat()


class WeeklyUpdates:
    def __init__(self, root=None, clock=utcnow, cooldown=900):
        self.root = Path(root or os.environ.get("ANSEL_UPDATE_DIR", PROJECT / "artifacts/weekly_updates")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "state.sqlite3"
        self.clock, self.cooldown = clock, cooldown
        self.stop = threading.Event()
        self.thread = None
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, period TEXT NOT NULL, status TEXT NOT NULL,
                    created REAL NOT NULL, finished REAL, stage TEXT, error TEXT,
                    trigger TEXT, result TEXT);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs((1))
                    WHERE status IN ('queued','running');
                CREATE TABLE IF NOT EXISTS publication (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    period TEXT NOT NULL, path TEXT NOT NULL, manifest TEXT NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=20000")
        return db

    def published(self):
        with self.connect() as db:
            row = db.execute("SELECT * FROM publication WHERE singleton=1").fetchone()
        if row is None:
            return None
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root / "versions"):
            raise ValueError("Invalid publication path")
        return {"period": row["period"], "path": path, "manifest": json.loads(row["manifest"])}

    def status(self, component="all"):
        period = latest_period(self.clock())
        publication = self.published()
        manifest = publication["manifest"] if publication else {}
        data_current = bool(publication and publication["period"] >= period and
                            manifest.get("pipeline_revision") == PIPELINE_REVISION and
                            all((publication["path"] / f).is_file() for f in DATA_FILES))
        report_current = bool(data_current and manifest.get("report_status") == "complete" and
                              (publication["path"] / "figures/weekly_report.md").is_file())
        current = report_current if component in ("report", "all") else data_current
        with self.connect() as db:
            active = db.execute("SELECT * FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
            last = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 1").fetchone()
        last = dict(last) if last else None
        retry_at = ((last["finished"] or 0)+self.cooldown
                    if last and last["period"] == period and last["status"] in ("failed", "partial") else 0)
        remaining = max(0, int(retry_at-self.clock().timestamp()))
        key_configured = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        outbound_enabled = os.environ.get("ANSEL_REPORT_OUTBOUND_ENABLED", "0") == "1"
        key_ready = key_configured and outbound_enabled
        if component == "all" and not key_ready:
            current = data_current
        report_needed = component in ("all", "report") and not report_current
        report_blocked = data_current and report_needed and not key_ready
        can_update = not current and not active and not remaining and not report_blocked
        if component == "all":
            can_update = (not data_current or (not report_current and key_ready)) and not active and not remaining
        return {"target_period": period, "data_period": publication["period"] if publication else None,
                "coverage_url": (f"/api/updates/versions/{publication['path'].name}/coverage.csv" if publication else None),
                "generated_at": manifest.get("generated_at"), "data_current": data_current,
                "report_current": report_current, "is_current": current,
                "running": active is not None, "job": dict(active) if active else None,
                "last_job": last, "retry_after_seconds": remaining,
                "can_update": bool(can_update), "report_configured": key_configured,
                "report_outbound_enabled": outbound_enabled,
                "coverage": manifest.get("coverage"),
                "report_error": ("AI 周报外部连接待管理员授权" if key_configured and not outbound_enabled
                                 else manifest.get("report_error")),
                "schedule": "每周四北京时间09:00起更新（UTC周三完整收盘后一小时）；停机后启动补更"}

    def request(self, trigger="manual", component="all"):
        # All endpoint aliases share this transaction. Caller-supplied force flags
        # never bypass freshness, cooldown or the unique active-job constraint.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self.status(component)
            if trigger == "scheduled":
                failures = db.execute("SELECT count(*) FROM jobs WHERE period=? AND status IN ('failed','partial')",
                                      (state["target_period"],)).fetchone()[0]
                if failures >= 3:
                    return {"status": "retry_limit", **state}
            if state["running"]:
                return {"status": "already_running", "job_id": state["job"]["id"], **state}
            if not state["can_update"]:
                code = ("up_to_date" if state["is_current"] else "cooldown" if state["retry_after_seconds"]
                        else "not_configured")
                return {"status": code, **state}
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,period,status,created,stage,trigger) VALUES(?,?,?,?,?,?)",
                       (job_id, state["target_period"], "queued", self.clock().timestamp(), "等待执行", trigger))
        return {"status": "started", "job_id": job_id, "target_period": state["target_period"]}

    def progress(self, job_id, stage):
        with self.connect() as db:
            db.execute("UPDATE jobs SET stage=? WHERE id=?", (stage, job_id))

    def publish(self, period, directory, manifest):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(self.root / "versions"):
            raise ValueError("Publication must be under versions")
        if not (directory / "manifest.json").is_file():
            raise ValueError("Cannot publish an unfinished directory")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO publication VALUES(1,?,?,?)",
                       (period, str(directory.relative_to(self.root)), json.dumps(manifest, ensure_ascii=False)))

    def run_pending(self, runner=None):
        with (self.root / "worker.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            with self.connect() as db:
                # Owning this lock proves no other compliant worker is executing.
                # A remaining running job came from an interrupted process.
                interrupted = db.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
                for row in interrupted:
                    completed = self.root / f"{row['id']}.result.json"
                    if completed.is_file():
                        recovered = json.loads(completed.read_text())
                        db.execute("UPDATE jobs SET status=?,finished=?,result=?,error=? WHERE id=?",
                            ("succeeded" if recovered.get("report_status") == "complete" else "partial",
                             self.clock().timestamp(), json.dumps(recovered), recovered.get("report_error"), row["id"]))
                    else:
                        db.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?",
                                   (self.clock().timestamp(), "上次更新被中断，旧版本已保留", row["id"]))
                job = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if job is None:
                    return False
                job = dict(job)
                db.execute("UPDATE jobs SET status='running',stage='准备历史数据' WHERE id=?", (job["id"],))
            try:
                if runner:
                    result = runner(self, job)
                else:
                    log = self.root / f"{job['id']}.log"
                    with log.open("w") as stream:
                        process = subprocess.run([sys.executable, str(PROJECT / "weekly_data_pipeline.py"),
                            "--root", str(self.root), "--job", job["id"], "--period", job["period"]],
                            cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, timeout=2700,
                            pass_fds=(lock.fileno(),))
                    if process.returncode:
                        # Do not expose raw subprocess logs or credentials publicly.
                        raise RuntimeError("数据更新未完成；已保留上一版本，请查看服务端任务日志")
                    result = json.loads((self.root / f"{job['id']}.result.json").read_text())
                with self.connect() as db:
                    db.execute("UPDATE jobs SET status=?,finished=?,stage=?,result=?,error=? WHERE id=?",
                               ("succeeded" if result.get("report_status") == "complete" else "partial",
                                self.clock().timestamp(), "更新完成", json.dumps(result, ensure_ascii=False),
                                result.get("report_error"), job["id"]))
            except Exception as exc:
                with self.connect() as db:
                    db.execute("UPDATE jobs SET status='failed',finished=?,stage='更新失败',error=? WHERE id=?",
                               (self.clock().timestamp(), str(exc), job["id"]))
            return True

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop.clear()

        def loop():
            while not self.stop.is_set():
                try:
                    self.request("scheduled", "all")
                    self.run_pending()
                except Exception:
                    import logging
                    logging.exception("Weekly update scheduler failed")
                self.stop.wait(15)
        self.thread = threading.Thread(target=loop, name="weekly-data-updater", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)


_service = None


def get_updates():
    global _service
    if _service is None:
        _service = WeeklyUpdates()
    return _service


def data_path(relative, fallback=None):
    publication = get_updates().published()
    if publication:
        path = publication["path"] / relative
        if path.is_file():
            return path
    return Path(fallback) if fallback else PROJECT / relative
