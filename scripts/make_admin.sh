#!/bin/bash
# Create admin user - usage: ./make_admin.sh [email] [password]
# If no args, runs interactive Python script
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || . venv/bin/activate
python scripts/create_admin.py "$@"
