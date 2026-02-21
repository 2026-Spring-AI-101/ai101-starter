#!/usr/bin/env bash
set -euo pipefail


# Fix: remove Yarn apt repo (sometimes breaks apt-get update)
sudo rm -f /etc/apt/sources.list.d/yarn.list || true

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ffmpeg pkg-config build-essential \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libavfilter-dev libswscale-dev libswresample-dev

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt



