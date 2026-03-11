"""
DeliveryIQ Persistence Layer
SQLite-based storage for projects, risk snapshots, chat history, and agent reports.
DB stored at ~/.deliveryiq/deliveryiq.db — survives app restarts.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# ── DB location ────────────────────────────────────────────────────────────────
DB_DIR = Path.home() / ".deliveryiq"
DB_PATH = DB_DIR / "deliveryiq.db"


def _get_conn():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist yet."""
    conn = _get_conn()
    c = conn.cursor()

    # Projects table — one row per named project
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            config      TEXT NOT NULL  -- JSON blob of all project settings
        )
    """)

    # Risk snapshots — one row per analysis run, linked to project
    c.execute("""
        CREATE TABLE IF NOT EXISTS risk_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name    TEXT NOT NULL,
            captured_at     TEXT NOT NULL,
            week_number     INTEGER,
            risk_level      TEXT,
            health_score    REAL,
            rag_status      TEXT,
            confidence      REAL,
            budget_health   REAL,
            timeline_health REAL,
            scope_health    REAL,
            team_health     REAL,
            stakeholder_health REAL,
            config_snapshot TEXT  -- JSON of inputs at time of analysis
        )
    """)

    # Chat history — per project, per module
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            module       TEXT NOT NULL,  -- 'knowledge_base' | 'agents'
            role         TEXT NOT NULL,  -- 'user' | 'assistant'
            content      TEXT NOT NULL,
            timestamp    TEXT NOT NULL
        )
    """)

    # Agent reports — saved LangGraph outputs
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            report_type  TEXT NOT NULL,  -- 'delivery' | 'risk' | 'stakeholder' etc.
            content      TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            metadata     TEXT           -- JSON for extra fields
        )
    """)

    conn.commit()
    conn.close()


# ── Projects ───────────────────────────────────────────────────────────────────

def save_project(name: str, config: dict) -> bool:
    """Insert or update a project record."""
    try:
        conn = _get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO projects (name, created_at, updated_at, config)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                updated_at = excluded.updated_at,
                config     = excluded.config
        """, (name, now, now, json.dumps(config)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[persistence] save_project error: {e}")
        return False


def load_project(name: str) -> dict | None:
    """Load a project config by name. Returns None if not found."""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT config FROM projects WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        return json.loads(row["config"]) if row else None
    except Exception as e:
        print(f"[persistence] load_project error: {e}")
        return None


def list_projects() -> list[dict]:
    """Return all projects sorted by most recently updated."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT name, created_at, updated_at, config FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [
            {
                "name": r["name"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                **json.loads(r["config"])
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[persistence] list_projects error: {e}")
        return []


def delete_project(name: str) -> bool:
    """Delete a project and all its associated data."""
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        conn.execute("DELETE FROM risk_snapshots WHERE project_name = ?", (name,))
        conn.execute("DELETE FROM chat_history WHERE project_name = ?", (name,))
        conn.execute("DELETE FROM agent_reports WHERE project_name = ?", (name,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[persistence] delete_project error: {e}")
        return False


# ── Risk Snapshots ─────────────────────────────────────────────────────────────

def save_risk_snapshot(project_name: str, snapshot: dict) -> bool:
    """Save a risk analysis result for trend tracking."""
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO risk_snapshots (
                project_name, captured_at, week_number,
                risk_level, health_score, rag_status, confidence,
                budget_health, timeline_health, scope_health,
                team_health, stakeholder_health, config_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_name,
            datetime.now().isoformat(),
            snapshot.get("week_number"),
            snapshot.get("risk_level"),
            snapshot.get("health_score"),
            snapshot.get("rag_status"),
            snapshot.get("confidence"),
            snapshot.get("budget_health"),
            snapshot.get("timeline_health"),
            snapshot.get("scope_health"),
            snapshot.get("team_health"),
            snapshot.get("stakeholder_health"),
            json.dumps(snapshot.get("config", {}))
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[persistence] save_risk_snapshot error: {e}")
        return False


def get_risk_history(project_name: str, limit: int = 20) -> list[dict]:
    """Return risk snapshots for a project, newest first."""
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT * FROM risk_snapshots
            WHERE project_name = ?
            ORDER BY captured_at DESC
            LIMIT ?
        """, (project_name, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[persistence] get_risk_history error: {e}")
        return []


def get_risk_trend(project_name: str, weeks: int = 6) -> list[dict]:
    """Return the last N snapshots in chronological order for trend charts."""
    history = get_risk_history(project_name, limit=weeks)
    return list(reversed(history))


# ── Chat History ───────────────────────────────────────────────────────────────

def save_chat_message(project_name: str, module: str, role: str, content: str) -> bool:
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO chat_history (project_name, module, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (project_name, module, role, content, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[persistence] save_chat_message error: {e}")
        return False


def load_chat_history(project_name: str, module: str, limit: int = 100) -> list[dict]:
    """Load chat messages for a project+module, oldest first."""
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT role, content, timestamp FROM chat_history
            WHERE project_name = ? AND module = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (project_name, module, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[persistence] load_chat_history error: {e}")
        return []


def clear_chat_history(project_name: str, module: str) -> bool:
    try:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM chat_history WHERE project_name = ? AND module = ?",
            (project_name, module)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[persistence] clear_chat_history error: {e}")
        return False


# ── Agent Reports ──────────────────────────────────────────────────────────────

def save_agent_report(project_name: str, report_type: str, content: str, metadata: dict = None) -> bool:
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO agent_reports (project_name, report_type, content, generated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            project_name, report_type, content,
            datetime.now().isoformat(),
            json.dumps(metadata or {})
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[persistence] save_agent_report error: {e}")
        return False


def get_agent_reports(project_name: str, report_type: str = None, limit: int = 10) -> list[dict]:
    try:
        conn = _get_conn()
        if report_type:
            rows = conn.execute("""
                SELECT * FROM agent_reports
                WHERE project_name = ? AND report_type = ?
                ORDER BY generated_at DESC LIMIT ?
            """, (project_name, report_type, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM agent_reports
                WHERE project_name = ?
                ORDER BY generated_at DESC LIMIT ?
            """, (project_name, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[persistence] get_agent_reports error: {e}")
        return []


# ── Stats / Summary ────────────────────────────────────────────────────────────

def get_project_summary(project_name: str) -> dict:
    """Return a quick summary dict for the sidebar / home dashboard."""
    try:
        history = get_risk_history(project_name, limit=1)
        trend = get_risk_trend(project_name, weeks=6)
        chat_kb = load_chat_history(project_name, "knowledge_base", limit=1000)
        chat_ag = load_chat_history(project_name, "agents", limit=1000)
        reports = get_agent_reports(project_name, limit=1000)

        latest = history[0] if history else {}

        # Compute trend direction
        trend_direction = "stable"
        if len(trend) >= 2:
            delta = (trend[-1].get("health_score") or 0) - (trend[0].get("health_score") or 0)
            if delta > 5:
                trend_direction = "improving"
            elif delta < -5:
                trend_direction = "declining"

        return {
            "latest_risk": latest.get("risk_level", "Unknown"),
            "latest_health": latest.get("health_score", 0),
            "latest_rag": latest.get("rag_status", "Unknown"),
            "snapshots_count": len(get_risk_history(project_name, limit=1000)),
            "trend_direction": trend_direction,
            "trend_data": trend,
            "chat_messages_kb": len(chat_kb),
            "chat_messages_agents": len(chat_ag),
            "reports_count": len(reports),
            "last_updated": latest.get("captured_at", "Never"),
        }
    except Exception as e:
        print(f"[persistence] get_project_summary error: {e}")
        return {}


# Initialise on import
init_db()
