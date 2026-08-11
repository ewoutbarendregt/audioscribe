#!/usr/bin/env python3
"""
Audioscribe authentication — email one-time-code (OTP) login.

Self-contained by design: Audioscribe is *hosted* under trustable.nl/projects/audioscribe
but has no other relationship to it, so it keeps its own user gate rather than reusing
trustable's session cookie or admin allowlist. That keeps the app portable.

Flow:
    request-otp  → 6-digit code emailed to an allow-listed address (hash stored)
    verify-otp   → code checked, single-use, exchanged for a session token
    session      → httpOnly cookie, token stored hashed, TTL in days

Nothing is stored in plaintext: both OTP codes and session tokens are kept as sha256
hashes, so a leaked database cannot be replayed.

Environment (set in /opt/trustable/audioscribe.env on each VPS):
    DATA_DIR          - directory for the SQLite file (default: /data)
    COOKIE_SECURE     - "false" only for plain-HTTP local dev (default: true)
    ALLOWED_EMAILS    - comma-separated allowlist; empty = nobody can log in
    SMTP_HOST/PORT/USER/PASS - mail transport (Resend: smtp.resend.com/587/resend/<key>)
    MAIL_FROM         - sender address on a domain verified with the provider
    OTP_TTL_MINUTES   - code lifetime (default: 10)
    SESSION_TTL_DAYS  - session lifetime (default: 30)
    COOKIE_PATH       - cookie scope (default: /); prod uses /projects/audioscribe
"""

import hashlib
import logging
import os
import secrets
import smtplib
import sqlite3
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSION_COOKIE = "audioscribe_session"

OTP_TTL_SECONDS = int(os.environ.get("OTP_TTL_MINUTES", "10")) * 60
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_DAYS", "30")) * 24 * 3600
COOKIE_PATH = os.environ.get("COOKIE_PATH", "/")
# HTTPS-only by default. Set COOKIE_SECURE=false only for plain-HTTP local dev —
# both VPSes terminate TLS, so it must stay true in staging and prod.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() != "false"
MAX_OTP_ATTEMPTS = 5

# Codes are 6 digits — short enough to retype from a phone, and the attempt cap plus
# the 10-minute expiry keep the guess space effectively out of reach.
_OTP_DIGITS = 6


def _resolve_data_dir() -> Path:
    """DATA_DIR if writable, else ./data next to this file (local dev)."""
    preferred = Path(os.environ.get("DATA_DIR", "/data"))
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write-test"
        probe.touch()
        probe.unlink()
        return preferred
    except Exception:
        fallback = Path(__file__).parent / "data"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning("DATA_DIR %s not writable — using %s", preferred, fallback)
        return fallback


DATA_DIR = _resolve_data_dir()
DB_PATH = DATA_DIR / "audioscribe.db"


def allowed_emails() -> set:
    """Read the allowlist at call time so it can change without a code deploy."""
    raw = os.environ.get("ALLOWED_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def normalize_email(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if absent. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS otp_codes (
                email      TEXT PRIMARY KEY,
                code_hash  TEXT NOT NULL,
                expires_at REAL NOT NULL,
                attempts   INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                email      TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_email ON sessions(email);
            """
        )
    logger.info("Auth database ready at %s", DB_PATH)


def purge_expired() -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute("DELETE FROM otp_codes WHERE expires_at < ?", (now,))
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))


# ---------------------------------------------------------------------------
# One-time codes
# ---------------------------------------------------------------------------
def create_otp(email: str) -> str:
    """Generate a code for `email`, store its hash, and return the plaintext.

    Any previous code for the address is replaced, so requesting a new code always
    invalidates the old one.
    """
    code = "".join(str(secrets.randbelow(10)) for _ in range(_OTP_DIGITS))
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO otp_codes (email, code_hash, expires_at, attempts, created_at)"
            " VALUES (?, ?, ?, 0, ?)",
            (email, sha256_hex(code), now + OTP_TTL_SECONDS, now),
        )
    return code


def verify_otp(email: str, code: str) -> bool:
    """Check a code. Consumes it on success; counts the attempt on failure."""
    now = time.time()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM otp_codes WHERE email = ?", (email,)).fetchone()
        if row is None:
            return False
        if row["expires_at"] < now or row["attempts"] >= MAX_OTP_ATTEMPTS:
            conn.execute("DELETE FROM otp_codes WHERE email = ?", (email,))
            return False
        if not secrets.compare_digest(row["code_hash"], sha256_hex(code)):
            conn.execute("UPDATE otp_codes SET attempts = attempts + 1 WHERE email = ?", (email,))
            return False
        conn.execute("DELETE FROM otp_codes WHERE email = ?", (email,))   # single use
        return True


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def create_session(email: str) -> str:
    """Create a session for `email` and return the plaintext token for the cookie."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, email, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (sha256_hex(token), email, now + SESSION_TTL_SECONDS, now),
        )
    return token


def session_email(token: Optional[str]) -> Optional[str]:
    """Return the email behind a session token, or None if absent/expired."""
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT email, expires_at FROM sessions WHERE token_hash = ?", (sha256_hex(token),)
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < time.time():
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (sha256_hex(token),))
            return None
        return row["email"]


def delete_session(token: Optional[str]) -> None:
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (sha256_hex(token),))


# ---------------------------------------------------------------------------
# Mail
# ---------------------------------------------------------------------------
def send_otp_email(email: str, code: str) -> None:
    """Email a login code. With no SMTP_HOST configured, log it instead (local dev).

    Blocking (smtplib) — callers should run this via asyncio.to_thread.
    """
    ttl_min = OTP_TTL_SECONDS // 60
    host = os.environ.get("SMTP_HOST")
    if not host:
        logger.warning("SMTP not configured — login code for %s is %s", email, code)
        return

    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    sender = os.environ.get("MAIL_FROM", "audioscribe@trustable.nl")

    msg = EmailMessage()
    msg["Subject"] = f"{code} is your Audioscribe login code"
    msg["From"] = sender
    msg["To"] = email
    msg.set_content(
        f"Your Audioscribe login code is {code}\n\n"
        f"It expires in {ttl_min} minutes and can be used once.\n\n"
        "If you didn't ask to sign in, you can ignore this email."
    )
    msg.add_alternative(
        f"""<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
background:#faf7f2;padding:32px;color:#211c16;">
  <div style="max-width:440px;margin:0 auto;background:#fff;border:1px solid #eadfcc;
border-radius:16px;padding:32px;text-align:center;">
    <div style="font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
color:#b4823a;">Audioscribe</div>
    <h1 style="font-size:20px;margin:14px 0 6px;">Your login code</h1>
    <p style="color:#6d6355;font-size:14px;margin:0 0 22px;">
      Expires in {ttl_min} minutes. Can be used once.</p>
    <div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:34px;
font-weight:700;letter-spacing:.18em;background:#fbf3e6;border:1px solid #f0dcbb;
border-radius:12px;padding:16px;">{code}</div>
    <p style="color:#9a9083;font-size:12px;margin:22px 0 0;">
      If you didn't ask to sign in, you can ignore this email.</p>
  </div>
</body></html>""",
        subtype="html",
    )

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        if user or password:
            server.login(user, password)
        server.send_message(msg)
    logger.info("Login code sent to %s", email)
