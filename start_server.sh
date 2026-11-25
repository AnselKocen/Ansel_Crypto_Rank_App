#!/bin/bash
# Ansel Crypto Rank - Server Start Script

cd "$(dirname "$0")"

# Fix sqlite3 library path issue
export DYLD_LIBRARY_PATH="/usr/local/opt/sqlite/lib:$DYLD_LIBRARY_PATH"

# The old .venv may contain a Python symlink from another computer.
if [ -x ".replay-venv/bin/python" ]; then
    TASK_PYTHON="$PWD/.replay-venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    TASK_PYTHON="$PWD/.venv/bin/python"
else
    echo "No working project Python environment found."
    exit 1
fi
export NLTK_DATA="$PWD/artifacts/historical_replay/nltk_data"
export MPLCONFIGDIR="$PWD/artifacts/historical_replay/mpl_config"

echo "🚀 Starting Ansel Crypto Rank Server..."
echo "📍 Local: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start server
"$TASK_PYTHON" -m uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload
