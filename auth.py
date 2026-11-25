"""
ResearchAI — Authentication Module

Handles user registration, login, logout, password hashing (PBKDF2-HMAC-SHA256),
and Streamlit session management. Every user's data is isolated by user_id.
"""

import hashlib
import os
import re
from datetime import datetime, timezone
from functools import wraps

from database.database import get_session
from database.models import User


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------

_HASH_ITERATIONS = 600_000  # OWASP-recommended minimum for PBKDF2-SHA256
_SALT_BYTES = 32
_KEY_LENGTH = 32  # 256-bit derived key


def hash_password(password: str) -> tuple[str, str]:
    """Hash *password* with a random salt.

    Returns (hash_hex, salt_hex).
    """
    salt = os.urandom(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _HASH_ITERATIONS,
        dklen=_KEY_LENGTH,
    )
    return dk.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Return True when *password* matches *stored_hash* / *salt*."""
    try:
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            _HASH_ITERATIONS,
            dklen=_KEY_LENGTH,
        )
        # Constant-time comparison to prevent timing attacks
        return hmac_compare(dk.hex(), stored_hash)
    except (ValueError, TypeError):
        return False


def hmac_compare(a: str, b: str) -> bool:
    """Constant-time string comparison."""
    import hmac as _hmac
    return _hmac.compare_digest(a, b)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_registration(username: str, email: str, password: str) -> None:
    """Raise ValueError with a user-friendly message on bad input."""
    if not username or len(username.strip()) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if not email or not _EMAIL_RE.match(email.strip()):
        raise ValueError("Please enter a valid email address.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")


# ---------------------------------------------------------------------------
# Registration & authentication
# ---------------------------------------------------------------------------

def register_user(username: str, email: str, password: str) -> User:
    """Create a new user account.

    Returns the created `User` on success.
    Raises `ValueError` with a descriptive message on validation or
    uniqueness failure.
    """
    username = username.strip()
    email = email.strip().lower()

    _validate_registration(username, email, password)

    with get_session() as session:
        # Check uniqueness
        if session.query(User).filter(User.username == username).first():
            raise ValueError("Username already taken.")
        if session.query(User).filter(User.email == email).first():
            raise ValueError("Email already registered.")

        pw_hash, salt = hash_password(password)

        user = User(
            username=username,
            email=email,
            password_hash=pw_hash,
            salt=salt,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def authenticate_user(username: str, password: str) -> User | None:
    """Verify credentials and return the `User`, or None on failure.

    On success the user's `last_login` timestamp is updated.
    """
    username = username.strip()

    with get_session() as session:
        user = session.query(User).filter(User.username == username).first()
        if user is None:
            return None
        if not verify_password(password, user.password_hash, user.salt):
            return None

        user.last_login = datetime.now(timezone.utc)
        session.commit()
        session.refresh(user)
        return user


def reset_password(username: str, email: str, new_password: str) -> bool:
    """Reset a user's password after verifying username + email match.

    Returns True on success, False if the user is not found.
    Raises ValueError if the new password is too short.
    """
    if not new_password or len(new_password) < 6:
        raise ValueError("New password must be at least 6 characters.")

    username = username.strip()
    email = email.strip().lower()

    with get_session() as session:
        user = (
            session.query(User)
            .filter(User.username == username, User.email == email)
            .first()
        )
        if user is None:
            return False

        pw_hash, salt = hash_password(new_password)
        user.password_hash = pw_hash
        user.salt = salt
        session.commit()
        return True


# ---------------------------------------------------------------------------
# Session helpers (Streamlit)
# ---------------------------------------------------------------------------

def _get_st():
    """Lazy-import streamlit so this module is testable without it."""
    import streamlit as st
    return st


def login_user(user: User) -> None:
    """Store authentication state in Streamlit session_state."""
    st = _get_st()
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user.id
    st.session_state["username"] = user.username


def check_session() -> dict | None:
    """Return current user info dict from session_state, or None."""
    st = _get_st()
    if st.session_state.get("authenticated"):
        return {
            "user_id": st.session_state["user_id"],
            "username": st.session_state["username"],
        }
    return None


def get_current_user_id() -> int | None:
    """Shortcut: return the logged-in user's ID, or None."""
    info = check_session()
    return info["user_id"] if info else None


def require_auth():
    """Guard that stops the page and shows login prompt if not authenticated.

    Usage at the top of a Streamlit page function:
        require_auth()
        # ... rest of the page
    If the user is not authenticated, st.stop() is called, halting execution.
    """
    st = _get_st()
    if not st.session_state.get("authenticated"):
        st.warning("Please log in to access this page.")
        st.stop()


def logout() -> None:
    """Clear all authentication state from session_state."""
    st = _get_st()
    keys_to_clear = [
        "authenticated",
        "user_id",
        "username",
        "conversation_id",
        "chat_history",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
