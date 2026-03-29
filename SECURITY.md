# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅        |
| < 2.0   | ❌        |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email **security@sqlagent.ai** with:

1. A description of the vulnerability
2. Steps to reproduce (include minimal code / curl commands)
3. Potential impact assessment
4. Any suggested fix (optional but appreciated)

You will receive an acknowledgement within **48 hours** and a resolution timeline within **7 days**.

We follow coordinated disclosure — we ask that you give us 90 days to patch before publishing.

## Security design notes

- **PolicyGateway** — all SQL is checked against a deterministic allowlist before execution. No LLM calls involved. Cannot be hallucinated around.
- **Auth is opt-in** — `auth_enabled=False` by default for local use. Enable for multi-user deployments.
- **JWT secrets** — stored at `~/.sqlagent/.jwt_secret` with `0600` permissions. Override with `SQLAGENT_JWT_SECRET` env var.
- **CORS** — defaults to deny-all. Set `SQLAGENT_CORS_ORIGINS` for cross-origin access.
- **No telemetry by default** — OTel exports to console only unless `SQLAGENT_OTEL_ENDPOINT` is set.
