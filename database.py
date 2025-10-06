import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = "contact_data.db"
DB_PATH = Path(__file__).parent / DB_FILE

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Table for storing full AI JSON output
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_jsons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id INTEGER,
            ai_json TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (extraction_id) REFERENCES extractions (id)
        )
    ''')
    # Migration: Add new columns if missing
    new_json_columns = [
        ("created_at", "TEXT"),
        ("updated_at", "TEXT")
    ]
    for col, coltype in new_json_columns:
        cursor.execute("PRAGMA table_info(ai_jsons)")
        columns = [row[1] for row in cursor.fetchall()]
        if col not in columns:
            cursor.execute(f"ALTER TABLE ai_jsons ADD COLUMN {col} {coltype}")
    # --- Migration: Add new columns to extractions if missing ---
    new_columns = [
        ("primary_company", "TEXT"),
        ("hiring_departments", "TEXT"),
        ("important_notes", "TEXT"),
        ("created_at", "TEXT"),
        ("updated_at", "TEXT")
    ]
    for col, coltype in new_columns:
        cursor.execute(f"PRAGMA table_info(extractions)")
        columns = [row[1] for row in cursor.fetchall()]
        if col not in columns:
            cursor.execute(f"ALTER TABLE extractions ADD COLUMN {col} {coltype}")
    
    # Extraction batch table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_timestamp TEXT NOT NULL,
            source_file TEXT,
            raw_text TEXT,
            llm_provider TEXT,
            llm_model TEXT,
            primary_company TEXT,
            hiring_departments TEXT,
            important_notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')

    # Contacts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id INTEGER,
            name TEXT,
            title TEXT,
            company TEXT,
            email TEXT,
            phone TEXT,
            linkedin TEXT,
            department TEXT,
            confidence TEXT,
            notes TEXT,
            source TEXT,
            hiring_for_role TEXT,
            yoe TEXT,
            tech_stack TEXT,
            location TEXT,
            FOREIGN KEY (extraction_id) REFERENCES extractions (id)
        )
    ''')
    # Migration: Add new columns if missing
    new_contact_columns = [
        ("hiring_for_role", "TEXT"),
        ("yoe", "TEXT"),
        ("tech_stack", "TEXT"),
        ("location", "TEXT"),
        ("created_at", "TEXT"),
        ("updated_at", "TEXT")
    ]
    for col, coltype in new_contact_columns:
        cursor.execute("PRAGMA table_info(contacts)")
        columns = [row[1] for row in cursor.fetchall()]
        if col not in columns:
            cursor.execute(f"ALTER TABLE contacts ADD COLUMN {col} {coltype}")

    # Open roles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS open_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id INTEGER,
            role_title TEXT,
            experience_required TEXT,
            FOREIGN KEY (extraction_id) REFERENCES extractions (id)
        )
    ''')

    # Application instructions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS application_instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id INTEGER,
            method TEXT,
            recipient_email TEXT,
            email_subject_format TEXT,
            FOREIGN KEY (extraction_id) REFERENCES extractions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_contacts_to_db(results: dict, extraction_timestamp: str, source_file: str | None = None):
    from datetime import datetime
    now = datetime.now().isoformat(timespec='seconds')
    # Save full AI JSON to ai_jsons table
    import json as _json

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get LLM info
    llm_provider = None
    llm_model = None
    if results.get("llm_enhanced"):
        llm_provider, _ = results["llm_enhanced"][0]
        llm_model = results["llm_enhanced"][0][0]

    # Extract new HR fields
    general_info = results.get("general_info", {})
    primary_company = general_info.get("primary_company", "")
    hiring_departments = ", ".join(general_info.get("hiring_departments", [])) if isinstance(general_info.get("hiring_departments", []), list) else general_info.get("hiring_departments", "")
    important_notes = ", ".join(results.get("important_notes", [])) if isinstance(results.get("important_notes", []), list) else results.get("important_notes", "")

    # Insert extraction record with new fields
    cursor.execute('''
        INSERT INTO extractions (extraction_timestamp, source_file, raw_text, llm_provider, llm_model, primary_company, hiring_departments, important_notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (extraction_timestamp, source_file, results.get("raw_text", ""), llm_provider, llm_model, primary_company, hiring_departments, important_notes, now, now))

    extraction_id = cursor.lastrowid

    # Insert AI JSON
    cursor.execute('''
        INSERT INTO ai_jsons (extraction_id, ai_json, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    ''', (extraction_id, _json.dumps(results, ensure_ascii=False), now, now))

    # Insert contacts
    contacts_to_insert = []
    for contact in results.get("contacts", []):
        contacts_to_insert.append((
            extraction_id,
            contact.get("name"),
            contact.get("title"),
            contact.get("company"),
            contact.get("email"),
            contact.get("phone"),
            contact.get("linkedin"),
            contact.get("department"),
            contact.get("confidence"),
            contact.get("notes"),
            contact.get("source"),
            contact.get("hiring_for_role", ""),
            contact.get("yoe", ""),
            contact.get("tech_stack", ""),
            contact.get("location", ""),
            now,
            now
        ))
    cursor.executemany('''
        INSERT INTO contacts (extraction_id, name, title, company, email, phone, linkedin, department, confidence, notes, source, hiring_for_role, yoe, tech_stack, location, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', contacts_to_insert)

    # Insert open_roles
    open_roles = results.get("open_roles", [])
    open_roles_to_insert = []
    for role in open_roles:
        open_roles_to_insert.append((
            extraction_id,
            role.get("role_title", ""),
            role.get("experience_required", "")
        ))
    if open_roles_to_insert:
        cursor.executemany('''
            INSERT INTO open_roles (extraction_id, role_title, experience_required)
            VALUES (?, ?, ?)
        ''', open_roles_to_insert)

    # Insert application_instructions
    app_inst = results.get("application_instructions", {})
    if app_inst:
        cursor.execute('''
            INSERT INTO application_instructions (extraction_id, method, recipient_email, email_subject_format)
            VALUES (?, ?, ?, ?)
        ''', (
            extraction_id,
            app_inst.get("method", ""),
            app_inst.get("recipient_email", ""),
            app_inst.get("email_subject_format", "")
        ))

    conn.commit()
    conn.close()
    return extraction_id

def get_all_contacts_df():
    """Retrieve all contacts from the database as a DataFrame."""
    conn = get_db_connection()
    # Join with extractions to include extracted raw text
    df = pd.read_sql_query(
        '''
        SELECT contacts.*, extractions.raw_text AS extracted_text
        FROM contacts
        LEFT JOIN extractions ON contacts.extraction_id = extractions.id
        ''', conn)
    conn.close()
    return df

def get_extraction_details(extraction_id: int):
    """Get details for a specific extraction."""
    conn = get_db_connection()
    extraction = conn.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    contacts = pd.read_sql_query("SELECT * FROM contacts WHERE extraction_id = ?", conn, params=(extraction_id,))
    conn.close()
    return extraction, contacts

def get_all_ai_jsons_df():
    """Retrieve all AI JSONs from the database as a DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query(
        '''
        SELECT ai_jsons.*, extractions.extraction_timestamp, extractions.source_file
        FROM ai_jsons
        LEFT JOIN extractions ON ai_jsons.extraction_id = extractions.id
        ORDER BY ai_jsons.id DESC
        ''', conn)
    conn.close()
    return df

def update_timestamp(table: str, record_id: int):
    """Update the updated_at column for a given table and record id."""
    from datetime import datetime
    now = datetime.now().isoformat(timespec='seconds')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE {table} SET updated_at = ? WHERE id = ?", (now, record_id))
    conn.commit()
    conn.close()

def create_update_triggers():
    """Create triggers to auto-update updated_at on row update for key tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for table in ["contacts", "extractions", "ai_jsons"]:
        cursor.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_updated_at_trigger
            AFTER UPDATE ON {table}
            FOR EACH ROW
            BEGIN
                UPDATE {table} SET updated_at = datetime('now') WHERE id = NEW.id;
            END;
        """)
    conn.commit()
    conn.close()

# Initialize the database and tables on first import
create_tables()
create_update_triggers()
