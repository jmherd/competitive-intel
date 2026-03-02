# reports.py
# Saves and retrieves competitive intelligence reports from SQLite.

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "data/reports.db"


def initialize_database():
    """Create database and tables if they don't exist."""
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id            TEXT PRIMARY KEY,
            company_name  TEXT NOT NULL,
            company_data  TEXT,
            competitor_data TEXT,
            news_data     TEXT,
            report_data   TEXT,
            created_at    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_report(report_id, company_name, results):
    """Save a complete intelligence report to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    def serialize(obj):
        """Convert Pydantic model to JSON string safely."""
        if obj is None:
            return None
        try:
            return obj.model_dump_json()
        except:
            return json.dumps(str(obj))

    cursor.execute("""
        INSERT OR REPLACE INTO reports
        (id, company_name, company_data, competitor_data, news_data, report_data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        company_name,
        serialize(results.get("company")),
        serialize(results.get("competitors")),
        serialize(results.get("news")),
        serialize(results.get("report")),
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()
    conn.close()


def get_all_reports():
    """Retrieve all saved reports, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, company_name, report_data, created_at
        FROM reports
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [{
        "id": row[0],
        "company_name": row[1],
        "report_data": row[2],
        "created_at": row[3]
    } for row in rows]


def get_report_by_id(report_id):
    """Retrieve a full report by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM reports WHERE id = ?",
        (report_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "company_name": row[1],
        "company_data": row[2],
        "competitor_data": row[3],
        "news_data": row[4],
        "report_data": row[5],
        "created_at": row[6]
    }


def delete_report(report_id):
    """Delete a report by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()