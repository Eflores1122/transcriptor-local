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

python3 -m venv .venv
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt
echo "Instalación terminada. Ejecuta ./run.sh para abrir la app."
