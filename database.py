import os
import sqlite3
import pandas as pd
from pathlib import Path

# Optional Supabase support (uses supabase-py). It's optional; if
# SUPABASE_URL and SUPABASE_KEY are provided via env vars and the
# package is installed, the code will write/read from Supabase instead
# of local SQLite. Creating tables in Supabase must be done separately
# (see SQL DDL in the comments below).
try:
    from supabase import create_client
    SUPABASE_PY_AVAILABLE = True
except Exception:
    create_client = None
    SUPABASE_PY_AVAILABLE = False


def _log_supabase_error(e, context: str = "Supabase operation"):
    """Print a helpful, decoded Supabase error message and traceback.

    Supabase client errors sometimes embed a bytes-encoded JSON string in a
    `details` field (e.g. b'{"message":"Invalid API key"}'). This helper
    attempts to surface that inner message for easier debugging.
    """
    import traceback
    import json
    import re

    print(f"{context} failed:")
    try:
        # If the exception is already a mapping-like object, pretty-print it.
        if isinstance(e, dict):
            try:
                print(json.dumps(e, indent=2, ensure_ascii=False))
            except Exception:
                print(repr(e))
            details = e.get("details")
            if details:
                # details is often a bytes repr: b'{...}'. Try to extract JSON.
                try:
                    if isinstance(details, (bytes, bytearray)):
                        s = details.decode("utf-8", errors="replace")
                    else:
                        s = str(details)
                        m = re.search(r"b'(.*)'", s)
                        if m:
                            s = m.group(1)
                    # Attempt to parse JSON from the inner string
                    try:
                        parsed = json.loads(s)
                        print("Details:", json.dumps(parsed, indent=2, ensure_ascii=False))
                    except Exception:
                        print("Details (raw):", s)
                except Exception:
                    print("Could not decode details field:", repr(details))
        else:
            # Fallback: print representation
            print(repr(e))
    except Exception as ex:
        print("Error while formatting Supabase exception:", repr(ex))
    traceback.print_exc()

DB_FILE = "contact_data.db"
DB_PATH = Path(__file__).parent / DB_FILE


def get_db_connection():
    """Create a local SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def supabase_configured():
    """Return True if SUPABASE is configured in the environment and client lib is available."""
    return SUPABASE_PY_AVAILABLE and bool(os.environ.get("SUPABASE_URL")) and bool(os.environ.get("SUPABASE_KEY"))


def get_supabase_client():
    """Return a Supabase client using SUPABASE_URL and SUPABASE_KEY env vars.

    Note: The Supabase tables must already exist. See the DDL comment below
    for the SQL to run in the Supabase SQL editor.
    """
    if not supabase_configured():
        raise RuntimeError("Supabase is not configured (set SUPABASE_URL and SUPABASE_KEY and install supabase-py)")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    return create_client(url, key)


def save_contacts_to_supabase(results: dict, extraction_timestamp: str, source_file: str | None = None):
    """Persist extraction results to Supabase tables. Returns extraction id."""
    sb = get_supabase_client()
    from datetime import datetime
    now = datetime.now().isoformat(timespec='seconds')

    # Prepare LLM info and general fields
    llm_provider = None
    llm_model = None
    if results.get("llm_enhanced"):
        llm_provider, _ = results["llm_enhanced"][0]
        llm_model = results["llm_enhanced"][0][0]

    general_info = results.get("general_info", {})
    primary_company = general_info.get("primary_company", "")
    hiring_departments = ", ".join(general_info.get("hiring_departments", [])) if isinstance(general_info.get("hiring_departments", []), list) else general_info.get("hiring_departments", "")
    important_notes = ", ".join(results.get("important_notes", [])) if isinstance(results.get("important_notes", []), list) else results.get("important_notes", "")

    # Insert extraction
    extraction_payload = {
        "extraction_timestamp": extraction_timestamp,
        "source_file": source_file,
        "raw_text": results.get("raw_text", ""),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "primary_company": primary_company,
        "hiring_departments": hiring_departments,
        "important_notes": important_notes,
    }
    res = sb.table("extractions").insert(extraction_payload).execute()
    if getattr(res, "data", None) is None:
        raise RuntimeError(f"Supabase insertion error: no data returned from insert. response: {res}")
    # Expect first returned row to contain id
    extraction_id = res.data[0].get("id") if res.data else None
    if extraction_id is None:
        raise RuntimeError(f"Supabase insertion error: could not obtain extraction id. response: {res}")

    # Insert ai_json
    res_json = sb.table("ai_jsons").insert({
        "extraction_id": extraction_id,
        "ai_json": results,
    }).execute()
    if getattr(res_json, "data", None) is None:
        raise RuntimeError(f"Supabase ai_json insert error: no data returned. response: {res_json}")

    # Insert contacts
    contacts = results.get("contacts", [])
    if contacts:
        contacts_payload = []
        for c in contacts:
            payload = {
                "extraction_id": extraction_id,
                "name": c.get("name"),
                "title": c.get("title"),
                "company": c.get("company"),
                "email": c.get("email"),
                "phone": c.get("phone"),
                "linkedin": c.get("linkedin"),
                "department": c.get("department"),
                "confidence": c.get("confidence"),
                "notes": c.get("notes"),
                "source": c.get("source"),
                "hiring_for_role": c.get("hiring_for_role", ""),
                "yoe": c.get("yoe", ""),
                "tech_stack": c.get("tech_stack", ""),
                "location": c.get("location", ""),
            }
            contacts_payload.append(payload)
        res_contacts = sb.table("contacts").insert(contacts_payload).execute()
        if getattr(res_contacts, "data", None) is None:
            raise RuntimeError(f"Supabase contacts insert error: no data returned. response: {res_contacts}")

    # Insert open_roles
    open_roles = results.get("open_roles", [])
    if open_roles:
        roles_payload = [{
            "extraction_id": extraction_id,
            "role_title": r.get("role_title", ""),
            "experience_required": r.get("experience_required", ""),
        } for r in open_roles]
        res_roles = sb.table("open_roles").insert(roles_payload).execute()
        if getattr(res_roles, "data", None) is None:
            raise RuntimeError(f"Supabase open_roles insert error: no data returned. response: {res_roles}")

    # Application instructions
    app_inst = results.get("application_instructions", {})
    if app_inst:
        res_app = sb.table("application_instructions").insert({
            "extraction_id": extraction_id,
            "method": app_inst.get("method", ""),
            "recipient_email": app_inst.get("recipient_email", ""),
            "email_subject_format": app_inst.get("email_subject_format", ""),
        }).execute()
        if getattr(res_app, "data", None) is None:
            raise RuntimeError(f"Supabase application_instructions insert error: no data returned. response: {res_app}")

    return extraction_id


def get_all_contacts_df_supabase():
    sb = get_supabase_client()
    # Fetch contacts with extraction info via RPC-like query (simple approach: fetch both and merge)
    contacts_res = sb.table("contacts").select("*").order("id", desc=True).execute()
    extractions_res = sb.table("extractions").select("id, raw_text").execute()
    contacts = contacts_res.data or []
    extras = {e["id"]: e for e in (extractions_res.data or [])}
    # attach extracted_text where possible
    for c in contacts:
        ex = extras.get(c.get("extraction_id"))
        c["extracted_text"] = ex.get("raw_text") if ex else None
    return pd.DataFrame(contacts)


def get_all_ai_jsons_df_supabase():
    sb = get_supabase_client()
    res = sb.table("ai_jsons").select("*, extractions(extraction_timestamp, source_file)").order("id", desc=True).execute()
    rows = res.data or []
    # Map nested extraction fields to top-level columns for compatibility
    normalized = []
    for r in rows:
        row = r.copy()
        extraction = row.pop("extractions", None)
        if extraction:
            row["extraction_timestamp"] = extraction.get("extraction_timestamp")
            row["source_file"] = extraction.get("source_file")
        normalized.append(row)
    return pd.DataFrame(normalized)


# ---------------------------------------------------------
# Supabase table DDL (run these in Supabase SQL editor once):
#
# -- extractions
# CREATE TABLE public.extractions (
#   id bigserial PRIMARY KEY,
#   extraction_timestamp text NOT NULL,
#   source_file text,
#   raw_text text,
#   llm_provider text,
#   llm_model text,
#   primary_company text,
#   hiring_departments text,
#   important_notes text,
#   created_at timestamptz DEFAULT now(),
#   updated_at timestamptz DEFAULT now()
# );
#
# -- contacts
# CREATE TABLE public.contacts (
#   id bigserial PRIMARY KEY,
#   extraction_id bigint REFERENCES public.extractions(id),
#   name text,
#   title text,
#   company text,
#   email text,
#   phone text,
#   linkedin text,
#   department text,
#   confidence text,
#   notes text,
#   source text,
#   hiring_for_role text,
#   yoe text,
#   tech_stack text,
#   location text,
#   created_at timestamptz DEFAULT now(),
#   updated_at timestamptz DEFAULT now()
# );
#
# -- ai_jsons
# CREATE TABLE public.ai_jsons (
#   id bigserial PRIMARY KEY,
#   extraction_id bigint REFERENCES public.extractions(id),
#   ai_json jsonb,
#   created_at timestamptz DEFAULT now(),
#   updated_at timestamptz DEFAULT now()
# );
#
# -- open_roles
# CREATE TABLE public.open_roles (
#   id bigserial PRIMARY KEY,
#   extraction_id bigint REFERENCES public.extractions(id),
#   role_title text,
#   experience_required text
# );
#
# -- application_instructions
# CREATE TABLE public.application_instructions (
#   id bigserial PRIMARY KEY,
#   extraction_id bigint REFERENCES public.extractions(id),
#   method text,
#   recipient_email text,
#   email_subject_format text
# );
# ---------------------------------------------------------

def create_tables():
    # Only create local SQLite tables here. If using Supabase, the
    # user should create tables with the DDL provided above in Supabase.
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
    # If Supabase is configured, try Supabase but fall back to SQLite on error
    if supabase_configured():
        try:
            return save_contacts_to_supabase(results, extraction_timestamp, source_file)
        except Exception as e:
            # Log the error and continue with local SQLite fallback
            _log_supabase_error(e, context="Supabase save")

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
    if supabase_configured():
        try:
            return get_all_contacts_df_supabase()
        except Exception as e:
            _log_supabase_error(e, context="Supabase read (get_all_contacts_df)")

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
    if supabase_configured():
        try:
            sb = get_supabase_client()
            extraction = sb.table("extractions").select("*").eq("id", extraction_id).execute()
            contacts = sb.table("contacts").select("*").eq("extraction_id", extraction_id).execute()
            return extraction.data[0] if extraction.data else None, pd.DataFrame(contacts.data)
        except Exception as e:
            _log_supabase_error(e, context="Supabase fetch (get_extraction_details)")

    conn = get_db_connection()
    extraction = conn.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    contacts = pd.read_sql_query("SELECT * FROM contacts WHERE extraction_id = ?", conn, params=(extraction_id,))
    conn.close()
    return extraction, contacts

def get_all_ai_jsons_df():
    """Retrieve all AI JSONs from the database as a DataFrame."""
    if supabase_configured():
        try:
            return get_all_ai_jsons_df_supabase()
        except Exception as e:
            _log_supabase_error(e, context="Supabase read (get_all_ai_jsons_df)")

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
