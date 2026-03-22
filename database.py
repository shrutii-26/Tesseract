import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "tesseract.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS commitments (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            owner TEXT DEFAULT 'You',
            urgency TEXT DEFAULT 'medium',
            context TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            days_stale INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS learning_items (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT DEFAULT '',
            type TEXT DEFAULT 'article',
            topic TEXT DEFAULT '',
            minutes INTEGER DEFAULT 15,
            status TEXT DEFAULT 'not-started',
            days_stale INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS guardian_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            channel TEXT DEFAULT 'unknown',
            decision TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS social_interests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interest TEXT UNIQUE NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS guardian_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT UNIQUE NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ── Commitments ───────────────────────────────────────────────────

def get_commitments():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM commitments ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_commitment(item: dict):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO commitments
        (id, task, owner, urgency, context, status, days_stale, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item["id"], item["task"], item.get("owner", "You"),
        item.get("urgency", "medium"), item.get("context", ""),
        item.get("status", "pending"), item.get("days_stale", 0),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def update_commitment_status(id: str, status: str):
    conn = get_conn()
    conn.execute(
        "UPDATE commitments SET status=?, updated_at=? WHERE id=?",
        (status, datetime.now().isoformat(), id)
    )
    conn.commit()
    conn.close()

def delete_commitment(id: str):
    conn = get_conn()
    conn.execute("DELETE FROM commitments WHERE id=?", (id,))
    conn.commit()
    conn.close()


# ── Learning Items ────────────────────────────────────────────────

def get_learning_items():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM learning_items ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_learning_item(item: dict):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO learning_items
        (id, title, url, type, topic, minutes, status, days_stale, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item["id"], item["title"], item.get("url", ""),
        item.get("type", "article"), item.get("topic", ""),
        item.get("minutes", 15), item.get("status", "not-started"),
        item.get("days_stale", 0), datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def update_learning_status(id: str, status: str):
    conn = get_conn()
    conn.execute(
        "UPDATE learning_items SET status=?, updated_at=? WHERE id=?",
        (status, datetime.now().isoformat(), id)
    )
    conn.commit()
    conn.close()

def delete_learning_item(id: str):
    conn = get_conn()
    conn.execute("DELETE FROM learning_items WHERE id=?", (id,))
    conn.commit()
    conn.close()


# ── Guardian History ──────────────────────────────────────────────

def add_guardian_history(sender: str, channel: str, decision: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO guardian_history (sender, channel, decision) VALUES (?, ?, ?)",
        (sender, channel, decision)
    )
    conn.commit()
    conn.close()

def get_guardian_history(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM guardian_history ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_guardian_patterns():
    conn = get_conn()
    rows = conn.execute("""
        SELECT sender,
               SUM(CASE WHEN decision='allow' THEN 1 ELSE 0 END) as allow_count,
               SUM(CASE WHEN decision='block' THEN 1 ELSE 0 END) as block_count
        FROM guardian_history
        GROUP BY sender
        ORDER BY (allow_count + block_count) DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Social Interests ──────────────────────────────────────────────

def get_interests():
    conn = get_conn()
    rows = conn.execute("SELECT interest FROM social_interests").fetchall()
    conn.close()
    return [r["interest"] for r in rows]

def save_interest(interest: str):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO social_interests (interest) VALUES (?)", (interest,))
    conn.commit()
    conn.close()

def delete_interest(interest: str):
    conn = get_conn()
    conn.execute("DELETE FROM social_interests WHERE interest=?", (interest,))
    conn.commit()
    conn.close()


# ── Guardian Rules ────────────────────────────────────────────────

def get_rules():
    conn = get_conn()
    rows = conn.execute("SELECT rule FROM guardian_rules").fetchall()
    conn.close()
    return [r["rule"] for r in rows]

def save_rule(rule: str):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO guardian_rules (rule) VALUES (?)", (rule,))
    conn.commit()
    conn.close()

def delete_rule(rule: str):
    conn = get_conn()
    conn.execute("DELETE FROM guardian_rules WHERE rule=?", (rule,))
    conn.commit()
    conn.close()


# Initialize DB on import
init_db()