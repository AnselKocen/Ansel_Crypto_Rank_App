#!/usr/bin/env python3
"""Compatibility entry point for the single shared weekly updater.

Existing --predict/--sentiment/--report flags remain accepted. Data-producing
steps share one atomic weekly snapshot rather than independently rewriting files.
"""
import argparse
import json
from weekly_update import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--sentiment', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    load_environment()
    from web.services.weekly_updates import get_updates
    service = get_updates()
    component = 'report' if args.report and not (args.predict or args.sentiment) else ('data' if args.predict or args.sentiment else 'all')
    print(json.dumps(service.request('legacy-cli', component), ensure_ascii=False), flush=True)
    service.run_pending()
    status = service.status(component)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status['is_current']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
