#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "No se encontró python3. Instala Python 3.9 o posterior."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "Falta FFmpeg. En macOS puedes instalarlo con: brew install ffmpeg"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

if ! ".venv/bin/python" -c "import faster_whisper, streamlit, reportlab" >/dev/null 2>&1; then
  ".venv/bin/python" -m pip install --upgrade pip
  ".venv/bin/python" -m pip install -r requirements.txt
fi
exec ".venv/bin/streamlit" run app.py --server.headless true --browser.gatherUsageStats false
