## Refactor Note: Imports are grouped into standard library, third-party, and local modules.
# Standard Library Imports
import base64
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import cast, Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present
"""Streamlit UI and orchestration for Job Contact Extractor.

This module contains only UI code and orchestration. All AI/LLM logic lives in
``llm.py`` and database helpers are in ``database.py``.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from loguru import logger
from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

from database import get_all_contacts_df, get_all_ai_jsons_df, supabase_configured
import os

# --- Logger Setup ---
LOG_PATH = Path(__file__).parent / "logs" / "job-helper.log"
logger.remove()
logger.add(LOG_PATH, rotation="10 MB", retention="10 days", level="INFO", enqueue=True)
logger.add(lambda msg: print(msg, end=""), level="WARNING")

# Import AI helpers
from llm import ContactExtractor
from utils import verify_passlock

# Session keys
SESS_KEY_EXTRACTOR = "extractor"
SESS_KEY_OPENAI_API = "openai_api_key"
SESS_KEY_GOOGLE_API = "google_api_key"
SESS_KEY_OPENAI_MODEL = "openai_model"
SESS_KEY_GEMINI_MODEL = "gemini_model"
SESS_KEY_CURRENT_RESULTS = "current_results"
SESS_KEY_TIMESTAMP = "current_timestamp"


def setup_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("API Keys")
        # Prefill from environment or .streamlit/secrets.toml if available
        from utils import get_openai_api_key, get_streamlit_secret
        pre_openai = st.session_state.get(SESS_KEY_OPENAI_API) or get_openai_api_key() or os.environ.get('OPENAI_API_KEY')
        pre_google = st.session_state.get(SESS_KEY_GOOGLE_API) or get_streamlit_secret('google_api_key') or os.environ.get('GOOGLE_API_KEY')
        st.text_input("OpenAI API Key", type="password", key=SESS_KEY_OPENAI_API, value=pre_openai)
        st.text_input("Google Gemini API Key", type="password", key=SESS_KEY_GOOGLE_API, value=pre_google)

        st.session_state[SESS_KEY_EXTRACTOR].set_api_keys(
            st.session_state.get(SESS_KEY_OPENAI_API),
            st.session_state.get(SESS_KEY_GOOGLE_API),
        )

        st.markdown("---")
        # Local override: choose which DB to use when running locally.
        # If FORCE_SUPABASE=1 is set in the environment (e.g. on Streamlit Cloud), Supabase will be used regardless.
        st.checkbox("Use Supabase for storage (local override)", key="use_supabase_checkbox")
        if os.environ.get("FORCE_SUPABASE") == "1":
            st.info("FORCE_SUPABASE=1 set: Supabase will be used regardless of local checkbox.")
        st.subheader("🧩 Model Selection")
        st.selectbox("OpenAI model", options=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5"], key=SESS_KEY_OPENAI_MODEL)
        st.selectbox("Gemini model", options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"], key=SESS_KEY_GEMINI_MODEL)


def handle_text_input() -> str:
    return st.text_area("Paste job posting, email signature, or contact information here:", height=300)


def handle_image_input() -> Optional[Image.Image]:
    st.markdown("**Upload or paste an image (from clipboard):**")
    img_file = st.file_uploader("Upload a screenshot or image", type=["png", "jpg", "jpeg"])
    paste_result = pbutton("📋 Paste an image from clipboard")
    if paste_result and paste_result.image_data is not None:
        img = paste_result.image_data
        if isinstance(img, Image.Image):
            st.image(img, caption="Pasted Image")
            return img
    if img_file:
        try:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image")
            return img
        except Exception as e:
            st.error(f"Error loading image: {e}")
    return None


def display_results():
    st.header("📊 Extraction Results")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Structured Contacts", "📝 Raw Text", "🤖 AI JSON", "🗃️ Database", "🧠 All AI JSONs"]) 

    results = st.session_state.get(SESS_KEY_CURRENT_RESULTS)
    if not results:
        st.info("👆 Provide input and click 'Extract Contacts' to see results here.")
        return

    timestamp = st.session_state.get(SESS_KEY_TIMESTAMP, "N/A")
    st.subheader(f"Results from: {timestamp}")

    with tab1:
        contacts = results.get("contacts")
        if contacts:
            df = pd.DataFrame(contacts)
            st.dataframe(df)
            st.download_button("📥 Download Contacts CSV", df.to_csv(index=False), f"contacts_{timestamp}.csv")
        else:
            st.info("No contacts were found.")

    with tab2:
        st.text_area("Raw input text:", value=results.get("raw_text", ""), height=300, disabled=True)

    with tab3:
        if results.get("llm_enhanced"):
            st.json(results["llm_enhanced"])
        else:
            st.info("No AI results to display.")

    with tab4:
        st.header("🗃️ All Contacts in Database")
        try:
            all_contacts_df = get_all_contacts_df()
            if not all_contacts_df.empty:
                st.dataframe(all_contacts_df)
                st.download_button("📥 Download All as CSV", all_contacts_df.to_csv(index=False), "all_contacts.csv")
            else:
                st.info("Database is empty.")
        except Exception as e:
            st.error(f"Could not load from database: {e}")

    with tab5:
        st.header("🧠 All AI JSONs in Database")
        try:
            ai_jsons_df = get_all_ai_jsons_df()
            if not ai_jsons_df.empty:
                st.dataframe(ai_jsons_df[["id", "extraction_id", "extraction_timestamp", "source_file"]])
                st.download_button("📥 Download All AI JSONs", ai_jsons_df.to_csv(index=False), "all_ai_jsons.csv")
                st.subheader("View AI JSON for Selected Row")
                selected = st.number_input("Select AI JSON row id", min_value=int(ai_jsons_df["id"].min()), max_value=int(ai_jsons_df["id"].max()), value=int(ai_jsons_df["id"].min()))
                row = ai_jsons_df[ai_jsons_df["id"] == selected]
                if not row.empty:
                    st.json(row.iloc[0]["ai_json"])
            else:
                st.info("No AI JSONs in database.")
        except Exception as e:
            st.error(f"Could not load AI JSONs from database: {e}")


def main():
    logger.debug("Starting main application loop.")
    st.set_page_config(page_title="Job Contact Extractor", page_icon="👔", layout="wide")
    st.title("👔 Job Search Contact Extractor")
    st.markdown("Extract HR/recruiter contact details from text or images using AI.")

    # --- Simple passlock login wall ---
    pass_lock_required = True
    if pass_lock_required:
        # If no passlock is configured, allow entry (dev convenience)
        from utils import get_passlock_raw
        stored = get_passlock_raw()
        if not stored:
            st.warning("No passlock configured in environment or .streamlit/secrets.toml. Proceeding without login.")
        else:
            if 'logged_in' not in st.session_state:
                st.session_state['logged_in'] = False
            if not st.session_state['logged_in']:
                st.header("🔒 Login")
                candidate = st.text_input("Enter passlock", type="password", key="__passlock_input")
                if st.button("Unlock"):
                    if verify_passlock(candidate):
                        st.session_state['logged_in'] = True
                        # Try to rerun the app; if the method isn't present, fall back to toggling a key and returning
                        try:
                            rerun = getattr(st, 'experimental_rerun', None)
                            if callable(rerun):
                                rerun()
                            else:
                                # Toggle a transient session key to force a re-render in some Streamlit versions
                                st.session_state['_login_toggled'] = not st.session_state.get('_login_toggled', False)
                                return
                        except Exception:
                            st.session_state['_login_toggled'] = not st.session_state.get('_login_toggled', False)
                            return
                    else:
                        st.error("Passlock incorrect. Check your .streamlit/secrets.toml or PASSLOCK env var.")
                # Stop further rendering until logged in
                if not st.session_state['logged_in']:
                    return

    if SESS_KEY_EXTRACTOR not in st.session_state:
        logger.info("Initializing ContactExtractor for the first time.")
        st.session_state[SESS_KEY_EXTRACTOR] = ContactExtractor()

    setup_sidebar()

    col1, col2 = st.columns(2)

    with col1:
        st.header("📝 Input")
        input_method = st.radio("Choose input method:", ["Text", "Image"], horizontal=True)

        text_content = ""
        uploaded_image = None

        if input_method == "Text":
            text_content = handle_text_input()
        else:
            uploaded_image = handle_image_input()

        st.markdown("---")
        ai_service = st.selectbox("AI Service:", ["openai", "gemini"])

        can_process = bool(text_content.strip()) or (uploaded_image is not None)
        if st.button("🔍 Extract Contacts", type="primary", disabled=not can_process):
            extractor: ContactExtractor = st.session_state[SESS_KEY_EXTRACTOR]
            openai_model = st.session_state.get(SESS_KEY_OPENAI_MODEL)
            gemini_model = st.session_state.get(SESS_KEY_GEMINI_MODEL)
            results = None

            with st.spinner("AI is processing..."):
                if uploaded_image:
                    results = extractor.process_image(uploaded_image, ai_service, openai_model, gemini_model)
                elif text_content:
                    results = extractor.process_text(text_content, ai_service, openai_model, gemini_model)

            if results:
                st.session_state[SESS_KEY_CURRENT_RESULTS] = results
                st.session_state[SESS_KEY_TIMESTAMP] = datetime.now().strftime("%Y%m%d_%H%M%S")
                extractor.save_results(results)
                st.success("✅ Extraction complete!")
                st.rerun()

    with col2:
        display_results()


if __name__ == "__main__":
    main()