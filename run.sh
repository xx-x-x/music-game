#!/bin/bash
cd "$(dirname "$0")"

# Use the shared venv from kinect-game if it exists
VENV="../kinect-game/venv"
if [ ! -d "$VENV" ]; then
    echo "Creating venv..."
    python3 -m venv venv
    VENV="venv"
    source "$VENV/bin/activate"
    pip install pygame numpy sounddevice miniaudio opencv-python mediapipe
else
    source "$VENV/bin/activate"
fi

exec python game.py "$@"
