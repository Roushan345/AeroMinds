import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "aerominds.db"

DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def init_db():
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT PRIMARY KEY,
            event_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source_file TEXT,
            evidence_url TEXT,
            zone_id TEXT,
            nearest_landmark TEXT,
            recommended_action TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


# Initialize database when module loads
init_db()


# =========================================================
# GENERATE INCIDENT ID
# =========================================================

def generate_incident_id():
    return f"AM-{uuid.uuid4().hex[:6].upper()}"


# =========================================================
# CREATE INCIDENT
# =========================================================

def create_incident(
    event_class,
    confidence,
    severity,
    source_file,
    evidence_url,
    zone_id="Zone A",
    nearest_landmark="Demo / Manually Configured",
    recommended_action="Review incident",
):

    incident_id = generate_incident_id()

    now = datetime.now().astimezone()

    timestamp = now.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    iso_time = now.isoformat()

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO incidents (
            incident_id,
            event_class,
            confidence,
            severity,
            timestamp,
            source_file,
            evidence_url,
            zone_id,
            nearest_landmark,
            recommended_action,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            event_class,
            float(confidence),
            severity,
            timestamp,
            source_file,
            evidence_url,
            zone_id,
            nearest_landmark,
            recommended_action,
            "PENDING",
            iso_time,
            iso_time,
        ),
    )

    conn.commit()
    conn.close()

    return get_incident(incident_id)


# =========================================================
# GET ONE INCIDENT
# =========================================================

def get_incident(incident_id):

    conn = get_connection()

    row = conn.execute(
        """
        SELECT *
        FROM incidents
        WHERE incident_id = ?
        """,
        (incident_id,),
    ).fetchone()

    conn.close()

    return dict(row) if row else None


# =========================================================
# GET ALL INCIDENTS
# =========================================================

def get_all_incidents():

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM incidents
        ORDER BY created_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# =========================================================
# UPDATE INCIDENT STATUS
# =========================================================

def update_incident_status(
    incident_id,
    new_status
):

    new_status = new_status.upper()

    allowed_statuses = {
        "PENDING",
        "ASSIGNED",
        "CLEARED",
    }

    if new_status not in allowed_statuses:
        return None

    incident = get_incident(
        incident_id
    )

    if incident is None:
        return None

    now = datetime.now().astimezone()

    conn = get_connection()

    conn.execute(
        """
        UPDATE incidents
        SET
            status = ?,
            updated_at = ?
        WHERE incident_id = ?
        """,
        (
            new_status,
            now.isoformat(),
            incident_id,
        ),
    )

    conn.commit()
    conn.close()

    return get_incident(
        incident_id
    )