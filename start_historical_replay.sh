#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
TASK_PYTHON="$TASK_ROOT/.replay-venv/bin/python"
if [ ! -x "$TASK_PYTHON" ]; then
    echo "Replay environment missing. Create it and install requirements_replay.txt first."
    exit 1
fi
if [ ! -f "$TASK_ROOT/artifacts/historical_replay/report/index.html" ]; then
    echo "Historical replay report has not been generated yet."
    exit 1
fi
echo "Historical replay: http://127.0.0.1:8012/"
exec "$TASK_PYTHON" -m http.server 8012 --bind 127.0.0.1 \
    --directory "$TASK_ROOT/artifacts/historical_replay/report"
