"""Authentication — JWT + Google OAuth + email magic link + AuthStore.

Auth is optional (auth_enabled=False by default). When disabled,
a default local user is returned for all requests.
"""

from __future__ import annotations

import os
import secrets
import time
import uuid
from datetime import datetime, timezone

import structlog

from sqlagent.models import User
from sqlagent.exceptions import InvalidToken

logger = structlog.get_logger()

# Default user when auth is disabled
LOCAL_USER = User(
    user_id="local",
    email="local@localhost",
    display_name="Local User",
    provider="local",
)


# ═══════════════════════════════════════════════════════════════════════════════
# JWT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _get_secret(config_secret: str = "") -> str:
    """Get JWT secret from config, env, or generate one."""
    if config_secret:
        return config_secret
    env_secret = os.environ.get("SQLAGENT_JWT_SECRET", "")
    if env_secret:
        return env_secret
    # Generate and persist
    secret_path = os.path.join(os.path.expanduser("~"), ".sqlagent", ".jwt_secret")
    if os.path.exists(secret_path):
        with open(secret_path) as f:
            return f.read().strip()
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    secret = secrets.token_hex(32)
    with open(secret_path, "w") as f:
        f.write(secret)
    # Restrict to owner read/write only — prevents other users on the system reading it
    os.chmod(secret_path, 0o600)
    logger.info("auth.jwt_secret_generated", path=secret_path)
    return secret


def create_token(user_id: str, secret: str = "", expires_hours: int = 72) -> str:
    """Create a JWT token for a user."""
    import jwt

    secret = _get_secret(secret)
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + (expires_hours * 3600),
        "iat": int(time.time()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, secret: str = "") -> dict:
    """Verify a JWT token. Returns {"user_id": ..., "exp": ...}."""
    import jwt

    secret = _get_secret(secret)
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise InvalidToken("Token expired")
    except jwt.InvalidTokenError as e:
        raise InvalidToken(f"Invalid token: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH
# ═══════════════════════════════════════════════════════════════════════════════


def google_auth_url(client_id: str, redirect_uri: str) -> str:
    """Generate Google OAuth redirect URL."""
    from urllib.parse import urlencode

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


async def google_callback(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict:
    """Exchange OAuth code for user info. Returns {email, name, picture}."""
    import httpx

    async with httpx.AsyncClient() as client:
        # Exchange code for tokens
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        # Get user info
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        user_resp.raise_for_status()
        return user_resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# EMAIL MAGIC LINK
# ═══════════════════════════════════════════════════════════════════════════════

_magic_codes: dict[str, tuple[str, float]] = {}  # email → (code, expires_at)


def send_magic_link(email: str) -> str:
    """Generate a 6-digit magic code. In production wire this to an email provider.

    The code is intentionally NOT logged at INFO level to avoid it appearing in
    log aggregation systems. It is returned to the caller so the server layer
    can send it via email — the HTTP response must NOT include it.
    """
    code = f"{secrets.randbelow(999999):06d}"
    _magic_codes[email.lower()] = (code, time.time() + 600)  # 10 min expiry
    logger.debug("auth.magic_link_generated", email=email)  # code excluded from log
    # TODO: replace with real email delivery (SendGrid, SES, Resend, etc.)
    return code


def verify_magic_link(email: str, code: str) -> bool:
    """Verify a magic code."""
    key = email.lower()
    if key not in _magic_codes:
        return False
    stored_code, expires_at = _magic_codes[key]
    if time.time() > expires_at:
        del _magic_codes[key]
        return False
    if stored_code != code:
        return False
    del _magic_codes[key]
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH STORE (SQLite)
# ═══════════════════════════════════════════════════════════════════════════════


class AuthStore:
    """User storage backed by SQLite."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".sqlagent", "auth.db")
        self._db_path = db_path
        self._initialized = False

    async def init(self) -> None:
        if self._initialized:
            return
        import aiosqlite

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    display_name TEXT DEFAULT '',
                    avatar_url TEXT DEFAULT '',
                    provider TEXT DEFAULT 'email',
                    created_at TEXT,
                    last_login TEXT
                )
            """)
            await db.commit()
        self._initialized = True

    async def get_or_create(
        self,
        email: str,
        display_name: str = "",
        avatar_url: str = "",
        provider: str = "email",
    ) -> User:
        """Get existing user or create new one."""
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = await cursor.fetchone()
            now = datetime.now(timezone.utc).isoformat()

            if row:
                await db.execute("UPDATE users SET last_login = ? WHERE email = ?", (now, email))
                await db.commit()
                return User(
                    user_id=row[0],
                    email=row[1],
                    display_name=row[2],
                    avatar_url=row[3],
                    provider=row[4],
                )

            user_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                (
                    user_id,
                    email,
                    display_name or email.split("@")[0],
                    avatar_url,
                    provider,
                    now,
                    now,
                ),
            )
            await db.commit()
            return User(
                user_id=user_id,
                email=email,
                display_name=display_name or email.split("@")[0],
                avatar_url=avatar_url,
                provider=provider,
            )

    async def get_by_id(self, user_id: str) -> User | None:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
            if not row:
                return None
            return User(
                user_id=row[0],
                email=row[1],
                display_name=row[2],
                avatar_url=row[3],
                provider=row[4],
            )
