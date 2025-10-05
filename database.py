
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
    # --- Migration: Add new columns to extractions if missing ---
    new_columns = [
        ("primary_company", "TEXT"),
        ("hiring_departments", "TEXT"),
        ("important_notes", "TEXT")
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
            important_notes TEXT
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
            FOREIGN KEY (extraction_id) REFERENCES extractions (id)
        )
    ''')

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
    """Save structured contacts to the database."""
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
        INSERT INTO extractions (extraction_timestamp, source_file, raw_text, llm_provider, llm_model, primary_company, hiring_departments, important_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (extraction_timestamp, source_file, results.get("raw_text", ""), llm_provider, llm_model, primary_company, hiring_departments, important_notes))

    extraction_id = cursor.lastrowid

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
            contact.get("source")
        ))
    cursor.executemany('''
        INSERT INTO contacts (extraction_id, name, title, company, email, phone, linkedin, department, confidence, notes, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    df = pd.read_sql_query("SELECT * FROM contacts", conn)
    conn.close()
    return df

def get_extraction_details(extraction_id: int):
    """Get details for a specific extraction."""
    conn = get_db_connection()
    extraction = conn.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    contacts = pd.read_sql_query("SELECT * FROM contacts WHERE extraction_id = ?", conn, params=(extraction_id,))
    conn.close()
    return extraction, contacts

# Initialize the database and tables on first import
create_tables()
