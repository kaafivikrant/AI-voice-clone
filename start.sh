#!/bin/bash
# Start the AI Agent Team for public access via ngrok.
# Usage: ./start.sh

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "  AI Agent Team — Production Launcher"
echo "========================================="
echo ""

# 1. Build frontend
echo "[1/3] Building frontend..."
cd frontend
npm run build --silent
cd ..
echo "  ✓ Frontend built → frontend/dist/"

# 2. Start backend (serves frontend + API + WebSocket on :8000)
echo "[2/3] Starting backend on port 8000..."
cd backend
source .venv312/bin/activate 2>/dev/null || true
python server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "  Waiting for backend..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "  ✓ Backend ready"
    break
  fi
  sleep 1
done

# 3. Start ngrok
echo "[3/3] Starting ngrok tunnel..."
echo ""
echo "========================================="
echo "  Share the ngrok URL with your friends!"
echo "========================================="
echo ""
ngrok http 8000

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null
