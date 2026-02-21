#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ffmpeg

python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "✅ setup done"
