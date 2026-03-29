#!/bin/bash
# sqlagent demo server
# Set your API key before running:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   OR export OPENAI_API_KEY="sk-..."

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "No API key set. Please export ANTHROPIC_API_KEY or OPENAI_API_KEY"
  exit 1
fi

MODEL="${SQLAGENT_MODEL:-claude-sonnet-4-6}"
DB="${SQLAGENT_DB:-sqlite:////tmp/sqlagent_demo.db}"
PORT="${SQLAGENT_PORT:-8080}"

exec "$(dirname "$0")/.venv/bin/python" -m sqlagent.cli serve --db "$DB" --model "$MODEL" --port "$PORT"
