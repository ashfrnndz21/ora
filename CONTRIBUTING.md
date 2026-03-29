# Contributing to ora

ora is open source and community-driven. We welcome contributions.

## Setup

```bash
git clone https://github.com/ora/ora
cd ora
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

## Three ways to contribute

### 1. Training packs
Share your NL→SQL pairs for your industry domain via QueryHub.
See `ora/hub.py` for the pack format.

### 2. Connectors
Add a new database connector. Follow the pattern in `ora/connectors/sql_connectors.py`.

### 3. Generator strategies
Add a new SQL generation strategy. Follow the pattern in `ora/generators.py`.

## Code style

- **ruff** for linting: `ruff check ora/`
- **pytest** for tests: `pytest tests/ -q`
- Python 3.11+ required
- All models are `@dataclass`
- All connectors/generators are Protocol-based (no inheritance)
- LLM calls go through `litellm` only — no direct openai/anthropic imports

## Rules

- `pytest tests/ -q` must pass before any PR
- PolicyGateway has zero LLM calls (deterministic only)
- QueryContext never shared across requests
- SOUL always wrapped in try/except (never breaks a query)
- UI is a single HTML file — no build step
