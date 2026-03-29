# ora examples

Runnable scripts to explore ora's capabilities.

## Setup

```bash
pip install ora-sql
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY="sk-proj-..."
```

## Scripts

| Script | What it shows |
|---|---|
| `quickstart.py` | Single-source query: connect → ask → get DataFrame |
| `multi_source.py` | Cross-database synthesis: two sources, one answer |
| `async_usage.py` | Async queries in FastAPI / Jupyter / asyncio |
| `training_feedback.py` | Teach ora from curated NL→SQL pairs |

## Run

```bash
# Simplest demo (uses bundled SQLite fixture)
python examples/quickstart.py

# Cross-source (two databases → one result)
python examples/multi_source.py

# Async (parallel queries, 3× faster)
python examples/async_usage.py

# Training (improve accuracy over time)
python examples/training_feedback.py
```

## Interactive workspace

```bash
ora serve --db sqlite:///tests/fixtures/northwind.db --port 8080
```

Open http://localhost:8080 → Beam.ai-inspired workspace with live execution traces.
