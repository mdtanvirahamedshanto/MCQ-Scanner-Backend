#!/bin/bash
# Run backend server
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || . venv/bin/activate
python run.py
