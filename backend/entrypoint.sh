#!/bin/sh
set -e

echo "Seeding database (skipped if already populated)..."
python seed.py

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
