"""
IBM DeliveryIQ — Session Manager
Persists login state across browser refreshes using cookies.
"""

import hashlib
import hmac
import json
import os
import time

# Secret key for signing session tokens — loaded from .env
def _get_secret():
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(env_path)
    except ImportError:
        pass
    return os.environ.get("DELIVERYIQ_SESSION_SECRET", "ibm-deliveryiq-default-secret-2024")

SESSION_COOKIE = "deliveryiq_session"
SESSION_TTL    = 60 * 60 * 8  # 8 hours


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def create_session_token(username: str, role: str) -> str:
    """Create a signed session token containing username + role + expiry."""
    secret = _get_secret()
    expiry = int(time.time()) + SESSION_TTL
    payload = json.dumps({"u": username, "r": role, "exp": expiry})
    import base64
    encoded = base64.b64encode(payload.encode()).decode()
    sig = _sign(encoded, secret)
    return f"{encoded}.{sig}"


def verify_session_token(token: str) -> dict | None:
    """
    Verify and decode a session token.
    Returns {"username": ..., "role": ...} or None if invalid/expired.
    """
    try:
        import base64
        secret = _get_secret()
        encoded, sig = token.rsplit(".", 1)
        # Verify signature
        if not hmac.compare_digest(_sign(encoded, secret), sig):
            return None
        payload = json.loads(base64.b64decode(encoded).decode())
        # Check expiry
        if payload.get("exp", 0) < int(time.time()):
            return None
        return {"username": payload["u"], "role": payload["r"]}
    except Exception:
        return None
