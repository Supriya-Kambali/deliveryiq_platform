"""
IBM DeliveryIQ — Project Registry
===================================
Persistent storage for team members and project context, keyed by IBM username.
Stored in module2_knowledge_rag/project_registry.json.
"""

import json
import os
from typing import List, Dict, Optional, Tuple

REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'module2_knowledge_rag', 'project_registry.json'
)


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_registry(data: dict):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ── TEAM MEMBERS ──────────────────────────────────────────────────

def get_team_members(username: str) -> List[Dict]:
    """Return list of team member dicts for this user."""
    registry = _load_registry()
    return registry.get(username, {}).get("team_members", [])


def save_team_members(username: str, members: List[Dict]):
    """Save team members list for this user."""
    registry = _load_registry()
    if username not in registry:
        registry[username] = {}
    registry[username]["team_members"] = members
    _save_registry(registry)


def format_team_for_prompt(members: List[Dict]) -> str:
    """
    Convert team list to a compact string for injection into agent prompts.
    Example: "Rahul Sharma (Senior PM), Asha Nair (Junior QA), Vikram Das (Mid-level Dev)"
    """
    if not members:
        return "No team members registered."
    parts = []
    for m in members:
        name = m.get("name", "Unknown").strip()
        role = m.get("role", "").strip()
        seniority = m.get("seniority", "").strip()
        email = m.get("email", "").strip()
        if name:
            label = f"{name} ({seniority} {role})" if seniority and role else name
            if email:
                label += f" <{email}>"
            parts.append(label)
    return ", ".join(parts) if parts else "No team members registered."


# ── PROJECTS ──────────────────────────────────────────────────────

def get_all_projects(username: str) -> Dict:
    """Return dict of all saved projects for this user."""
    registry = _load_registry()
    return registry.get(username, {}).get("projects", {})


def get_active_project(username: str) -> Optional[Dict]:
    """Return the currently active project dict, or None."""
    registry = _load_registry()
    user_data = registry.get(username, {})
    active_id = user_data.get("active_project_id")
    if active_id:
        return user_data.get("projects", {}).get(active_id)
    return None


def save_project(username: str, project_data: Dict) -> str:
    """
    Save or update a project. Returns the project_id.
    If project_data has no 'id', generates one and sets it as active.
    """
    import uuid
    registry = _load_registry()
    if username not in registry:
        registry[username] = {"team_members": [], "projects": {}, "active_project_id": None}

    project_id = project_data.get("id") or f"proj_{uuid.uuid4().hex[:8]}"
    project_data["id"] = project_id

    registry[username].setdefault("projects", {})[project_id] = project_data
    registry[username]["active_project_id"] = project_id
    _save_registry(registry)
    return project_id


def set_active_project(username: str, project_id: str):
    """Set which project is currently active."""
    registry = _load_registry()
    if username in registry:
        registry[username]["active_project_id"] = project_id
        _save_registry(registry)


def delete_project(username: str, project_id: str):
    """Delete a project. If it was active, clear the active pointer."""
    registry = _load_registry()
    user_data = registry.get(username, {})
    projects = user_data.get("projects", {})
    if project_id in projects:
        del projects[project_id]
        if user_data.get("active_project_id") == project_id:
            user_data["active_project_id"] = next(iter(projects), None)
        _save_registry(registry)


def get_project_names(username: str) -> List[Tuple[str, str]]:
    """Return list of (project_id, display_name) for dropdown."""
    projects = get_all_projects(username)
    result = []
    for pid, pdata in projects.items():
        name = pdata.get("project_name", pid)
        client = pdata.get("client_code", "")
        display = f"{name} ({client})" if client else name
        result.append((pid, display))
    return result
