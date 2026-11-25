#!/usr/bin/env python3
"""Run/catch up the same shared weekly job used by the website."""
import argparse
import json
from pathlib import Path
import os


def load_environment():
    path = Path(__file__).resolve().parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Request and execute one missing weekly update")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    load_environment()
    from web.services.weekly_updates import get_updates
    service = get_updates()
    if args.once:
        print(json.dumps(service.request("cli"), ensure_ascii=False), flush=True)
        service.run_pending()
    result = service.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.once and (not result["data_current"] or
                      (result["report_configured"] and result["report_outbound_enabled"]
                       and not result["report_current"])):
        raise SystemExit(1)
