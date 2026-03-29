# Changelog

All notable changes to **ora** (`pip install ora-sql`) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
ora uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Fixed CORS wildcard (`*`) + `allow_credentials=True` combination — now defaults to deny-all; set `SQLAGENT_CORS_ORIGINS` to opt in
- Fixed magic-link code returned in HTTP response — code no longer exposed to requester
- Added `chmod 0600` to JWT secret file to prevent other-user reads
- Added `.env` to `.gitignore` to prevent accidental credential commits

### Added
- Animated network graph background on workspace picker (data topology canvas)
- `--no-auth` / `auth_enabled=False` local mode — skip login screen entirely for OSS use
- `GET /health` now returns `auth_required` field for frontend routing
- `app.html` served with `no-cache, no-store` headers — browser always gets fresh version
- `SECURITY.md` — vulnerability disclosure policy
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `.github/workflows/ci.yml` — automated test, lint, security scan, Docker build on every PR
- `.github/dependabot.yml` — weekly automated dependency updates
- GitHub issue and PR templates

### Fixed
- 45+ bare `except Exception: pass` blocks replaced with `logger.debug(...)` calls
- `print()` statements in production code replaced with `structlog` calls
- Dockerfile now runs as non-root user and includes `HEALTHCHECK`
- `MagicLinkRequest` Pydantic forward-ref error causing `/openapi.json` 500
- Greeting endpoint used invalid model name `claude-haiku-3-5` — fixed to `claude-3-haiku-20240307`
- Greeting now always uses fastest available model regardless of query model setting

### Changed
- CORS default changed from `["*"]` to `[]` (deny-all) — **breaking for cross-origin setups**
- Dockerfile base installs `postgres,mysql` extras; use `--build-arg EXTRAS=all` for full suite
- JWT secret file permissions hardened to `0600`

## [2.0.0] — 2026-03-01

### Added
- Complete rewrite as agentic NL2SQL runtime
- LangGraph 7-stage pipeline (routing → schema pruning → retrieval → generation → execution → selection → audit)
- CHESS LSH schema pruning — 500 cols → ~8 relevant
- M-Schema serialization (XiYan format)
- CHASE-SQL parallel multi-generator with pairwise LLM selection
- ReFoRCE 3-stage self-correction loop
- SOUL user mental model inference
- Cross-source DuckDB synthesis for multi-DB queries
- PolicyGateway — deterministic SQL allowlist, zero LLM calls
- Beam.ai-inspired workspace UI (single HTML file, no build step)
- FastAPI server with 20+ endpoints + SSE streaming
- SQLite, PostgreSQL, DuckDB, MySQL, Snowflake connectors
- In-process Qdrant vector store for few-shot retrieval
- 3-tier memory: working → episodic → semantic
- QueryHub with 28 bundled NL→SQL packs
- AWS AgentCore adapter stubs
