"""
IBM DeliveryIQ — Authentication Module
=======================================
Handles user authentication and role management.
Reads credentials from module2_knowledge_rag/user.json.

Roles:
  manager  → Full platform access (all 5 modules)
  employee → Dashboard + Risk + Knowledge Base + AI Agents
  intern   → Dashboard + Risk Dashboard only
"""

import json
import os
from typing import Optional

# Path to the user database (relative to project root)
_USER_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "module2_knowledge_rag", "user.json"
)

# Role → allowed pages mapping
ROLE_PAGES: dict[str, list[str]] = {
    "manager":  ["🏠 Home", "📊 Risk Dashboard", "📚 Knowledge Base", "🤖 AI Agents", "🚀 MLOps & Deploy"],
    "employee": ["🏠 Home", "📊 Risk Dashboard", "📚 Knowledge Base", "🤖 AI Agents"],
    "intern":   ["🏠 Home", "📊 Risk Dashboard"],
}

ROLE_LABELS: dict[str, str] = {
    "manager":  "Delivery Manager",
    "employee": "Delivery Consultant",
    "intern":   "Intern",
}


def load_users() -> list[dict]:
    """Load user records from user.json. Returns empty list on error."""
    try:
        with open(_USER_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("users", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Validate credentials.

    Args:
        username: Plain-text username
        password: Plain-text password

    Returns:
        role string if credentials are valid, None otherwise.
        Passwords are never stored in session state or logs.
    """

    if not username or not password:
        return None

    username = username.strip()

    # Only allow IBM emails
    if not username.endswith("@ibm.com"):
        return None

    users = load_users()

    for user in users:
        if user.get("username") == username and user.get("password") == password:
            return user.get("role")

    return None


def get_user_role(username: str) -> Optional[str]:
    """Return the role for a given username, or None if not found."""
    users = load_users()
    for user in users:
        if user.get("username") == username.strip():
            return user.get("role")
    return None


def get_allowed_pages(role: str) -> list[str]:
    """Return the list of page keys accessible for a given role."""
    return ROLE_PAGES.get(role, ["🏠 Home"])


def get_role_label(role: str) -> str:
    """Return a human-readable label for the role."""
    return ROLE_LABELS.get(role, role.title())
