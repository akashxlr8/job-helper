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

# Third-Party Imports
import pandas as pd
import requests
import streamlit as st
from loguru import logger
from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

# Conditional Third-Party Imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

from openai import OpenAI

# Local Application Imports
from database import get_all_contacts_df, save_contacts_to_db

## Refactor Note: Constants are defined at the top for easy configuration and to avoid "magic strings".
# --- Configuration Constants ---
LOG_PATH = Path(__file__).parent / "logs" / "job-helper.log"
DATA_DIR = Path("extracted_contacts")
DEFAULT_PROMPT_TEMPLATE = """You are a meticulous HR data extraction bot. Your goal is to analyze job posting announcements and extract key information into a structured JSON format, following the example provided.

---
**EXAMPLE**

**TEXT:**
"Mohammed Imran [MI] - Head of Talent Acquisition & Talent Branding at JioHotstar / Disney Star... Yes, we are hiring across our Engineering, Security, Product, Design & Analytics team at JioHotstar!... Sr Staff Data Scientist [7+ yrs exp]... Staff/Sr Staff Backend Engineer [7+ yrs exp]... Pls apply on career site directly or email your resume: Subject line: 'Role' & 'Your current company' To: mohammed.imranullah@jiostar.com... Note: 1. We are not hiring Freshers..."

**JSON OUTPUT:**
{{
    "contacts": [{{
        "name": "Mohammed Imran",
        "title": "Head of Talent Acquisition & Talent Branding",
        "company": "JioHotstar / Disney Star",
        "email": "mohammed.imranullah@jiostar.com"
    }}],
    "general_info": {{
        "primary_company": "JioHotstar",
        "hiring_departments": ["Engineering", "Security", "Product", "Design", "Analytics"]
    }},
    "open_roles": [
        {{"role_title": "Sr Staff Data Scientist", "experience_required": "7+ yrs"}},
        {{"role_title": "Staff/Sr Staff Backend Engineer", "experience_required": "7+ yrs"}}
    ],
    "application_instructions": {{
        "method": "Apply on career site directly or email your resume",
        "recipient_email": "mohammed.imranullah@jiostar.com",
        "email_subject_format": "'Role' & 'Your current company'"
    }},
    "important_notes": ["We are not hiring Freshers at the moment"]
}}
---

**YOUR TASK**

Now, analyze the following text and provide the JSON output in the exact same format as the example. If a section or field is not present, use an empty string "" or an empty list [].

TEXT TO ANALYZE:
{text}
"""
VISION_PROMPT = "Extract all text. try to maintain the formatting as much as possible."

# Model Configurations
OPENAI_TEXT_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5"}
OPENAI_VISION_MODELS = {"gpt-4o", "gpt-4o-mini"}
DEFAULT_OPENAI_TEXT_MODEL = "gpt-4o"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4o"

GEMINI_TEXT_MODELS = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"}
GEMINI_VISION_MODELS = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"}
DEFAULT_GEMINI_TEXT_MODEL = "gemini-pro"
DEFAULT_GEMINI_VISION_MODEL = "gemini-pro-vision"

# Streamlit Session State Keys
SESS_KEY_EXTRACTOR = "extractor"
SESS_KEY_OPENAI_API = "openai_api_key"
SESS_KEY_GOOGLE_API = "google_api_key"
SESS_KEY_OPENAI_MODEL = "openai_model"
SESS_KEY_GEMINI_MODEL = "gemini_model"
SESS_KEY_CURRENT_RESULTS = "current_results"
SESS_KEY_PREVIOUS_RESULTS = "previous_results"
SESS_KEY_TIMESTAMP = "current_timestamp"

# --- Logger Setup ---
logger.remove()
logger.add(LOG_PATH, rotation="10 MB", retention="10 days", level="INFO", enqueue=True)

logger.add(lambda msg: print(msg, end=""), level="WARNING")


class ContactExtractor:
    """Extracts contacts from text or images using LLMs."""

    def __init__(self):
        self.openai_client: Optional[OpenAI] = None
        self.gemini_enabled: bool = False
        DATA_DIR.mkdir(exist_ok=True)

    def set_api_keys(self, openai_api_key: Optional[str], google_api_key: Optional[str]):
        """Initializes API clients based on provided keys."""
        # Configure OpenAI
        if openai_api_key and self.openai_client is None:
            logger.info("Configuring OpenAI API client.")
            self.openai_client = OpenAI(api_key=openai_api_key)
        elif not openai_api_key:
            self.openai_client = None

        # Configure Gemini
        if google_api_key and GENAI_AVAILABLE and not self.gemini_enabled:
            logger.info("Configuring Gemini API.")
            try:
                genai.configure(api_key=google_api_key)
                self.gemini_enabled = True
                logger.info("Gemini API configured successfully.")
            except Exception as e:
                self.gemini_enabled = False
                logger.error(f"Failed to configure Gemini API: {e}")
        elif not google_api_key:
            self.gemini_enabled = False

    # -------------------- Model Selection Helpers --------------------
    def _select_model(self, requested: Optional[str], purpose: str, provider: str) -> str:
        """Selects a valid model or falls back to a default."""
        if provider == "openai":
            valid_models = OPENAI_VISION_MODELS if purpose == "vision" else OPENAI_TEXT_MODELS
            default = DEFAULT_OPENAI_VISION_MODEL if purpose == "vision" else DEFAULT_OPENAI_TEXT_MODEL
        else:  # gemini
            valid_models = GEMINI_VISION_MODELS if purpose == "vision" else GEMINI_TEXT_MODELS
            default = DEFAULT_GEMINI_VISION_MODEL if purpose == "vision" else DEFAULT_GEMINI_TEXT_MODEL
        
        selected = requested if requested and requested in valid_models else default
        logger.debug(f"Model selection for {provider}/{purpose}: requested='{requested}', valid={valid_models}, selected='{selected}'")
        return selected

    # -------------------- Response Parsing Helpers --------------------
    def _clean_and_parse_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Cleans common artifacts from LLM JSON responses and parses the string."""
        try:
            logger.debug(f"Raw LLM response text: {response_text[:500]}")
            # Attempt to find the JSON block using regex, more robust than stripping
            match = re.search(r"\{[\s\S]*\}", response_text)
            if match:
                return json.loads(match.group(0))
            logger.warning("No JSON object found in the response text.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Raw response: {response_text!r}")
            return None

    def _create_fallback_json(self, text: str) -> Dict[str, Any]:
        """Creates a basic JSON structure from raw text when full parsing fails."""
        logger.debug(f"Fallback parsing triggered for text: {text[:500]}")
        name_match = re.search(r"Name:\s*(.+)", text, re.I)
        title_match = re.search(r"Title:\s*(.+)", text, re.I)
        company_match = re.search(r"Company:\s*(.+)", text, re.I)
        email_match = re.search(r"Email:\s*(\S+@\S+)", text, re.I)
        
        contact = {
            "name": (name_match.group(1) if name_match else "").strip(),
            "title": (title_match.group(1) if title_match else "").strip(),
            "company": (company_match.group(1) if company_match else "").strip(),
            "email": (email_match.group(1) if email_match else "").strip(),
            "phone": "", "linkedin": "", "department": "",
            "confidence": "low",
            "notes": "Fallback parsing due to AI response error.",
        }
        return {
            "contacts": [contact] if any(contact.values()) else [],
            "general_info": {},
            "refinements": {"corrections_made": "Fallback parsing applied."},
        }

    # -------------------- Core LLM Interaction --------------------
    def enhance_with_llm(self, text: str, provider: str, openai_model: Optional[str], gemini_model: Optional[str]) -> List[Tuple[str, Dict]]:
        """Orchestrates contact extraction using the specified LLM provider."""
        logger.debug(f"Enhance_with_llm called with text: {text[:200]}")
        logger.debug(f"Provider: {provider}, OpenAI model: {openai_model}, Gemini model: {gemini_model}")
        results = []
        prompt = DEFAULT_PROMPT_TEMPLATE.format(text=text)
        logger.debug(f"Prompt sent to LLM: {prompt[:500]}")

        if provider == "openai" and self.openai_client:
            model_name = self._select_model(openai_model, "text", "openai")
            logger.info(f"Requesting OpenAI enhancement with model: {model_name}")
            try:
                resp = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                    # temperature=0.1,
                )
                content = resp.choices[0].message.content or ""
                parsed_data = self._clean_and_parse_json(content)
                if parsed_data:
                    results.append((model_name, parsed_data))
                else:
                    results.append((model_name, self._create_fallback_json(content)))
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                st.warning(f"OpenAI API error: {e}")

        elif provider == "gemini" and self.gemini_enabled:
            model_name = self._select_model(gemini_model, "text", "gemini")
            logger.info(f"Requesting Gemini enhancement with model: {model_name}")
            try:
                _GM = getattr(genai, "GenerativeModel", None) if GENAI_AVAILABLE else None
                if _GM is None:
                    raise RuntimeError("Gemini GenerativeModel not available in this package version")
                model = _GM(model_name)
                resp = model.generate_content(prompt)
                content = getattr(resp, "text", "")
                parsed_data = self._clean_and_parse_json(content)
                if parsed_data:
                    results.append((model_name, parsed_data))
                else:
                    results.append((model_name, self._create_fallback_json(content)))
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                st.warning(f"Gemini API error: {e}")

        return results

    def extract_text_from_image(self, image: Image.Image, provider: str, openai_model: Optional[str], gemini_model: Optional[str]) -> Optional[str]:
        logger.debug(f"Image object: {image}")
        """Extracts text from an image using the specified vision model provider."""
        logger.info(f"Extracting text from image with provider: {provider}")
        
        if provider == "openai" and self.openai_client:
            model_name = self._select_model(openai_model, "vision", "openai")
            logger.info(f"Using OpenAI vision model: {model_name} (requested: {openai_model})")
            try:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                logger.info(f"Image encoded to base64, size: {len(base64_image)} characters")
                
                resp = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user","content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ]}],
                    max_tokens=4096,
                )
                content = resp.choices[0].message.content
                logger.info(f"OpenAI vision response received, content length: {len(content) if content else 0}")
                return content
            except Exception as e:
                logger.error(f"OpenAI Vision error: {e}")
                st.error(f"OpenAI Vision error: {e}")
                return None

        elif provider == "gemini" and self.gemini_enabled:
            model_name = self._select_model(gemini_model, "vision", "gemini")
            logger.info(f"Using Gemini vision model: {model_name} (requested: {gemini_model})")
            try:
                _GM = getattr(genai, "GenerativeModel", None) if GENAI_AVAILABLE else None
                if _GM is None:
                    raise RuntimeError("Gemini GenerativeModel not available in this package version")
                model = _GM(model_name)
                resp = model.generate_content([VISION_PROMPT, image])
                content = getattr(resp, "text", None)
                logger.info(f"Gemini vision response received, content length: {len(content) if content else 0}")
                return content
            except Exception as e:
                logger.error(f"Gemini Vision error: {e}")
                st.error(f"Gemini Vision error: {e}")
                return None
        
        else:
            logger.error(f"Vision extraction not available. Provider: {provider}, OpenAI client: {self.openai_client is not None}, Gemini enabled: {self.gemini_enabled}")
            st.error(f"Vision extraction not available for provider: {provider}")
            return None

    # -------------------- Pipelines --------------------
    def process_text(self, text: str, ai_service: str, openai_model: Optional[str], gemini_model: Optional[str]) -> Dict:
        """Processes raw text to extract contacts."""
        logger.debug(f"process_text input: {text[:500]}")
        logger.info(f"Starting text processing with service: {ai_service}")
        llm_results = self.enhance_with_llm(text, ai_service, openai_model, gemini_model)
        # Use the first LLM result for HR fields (assuming single model for now)
        hr_data = llm_results[0][1] if llm_results else {}
        structured_contacts = self.structure_final_results(llm_results)
        logger.info(f"Structured {len(structured_contacts)} contact(s).")
        return {
            "raw_text": text,
            "llm_enhanced": llm_results,
            "contacts": hr_data.get("contacts", []),
            "general_info": hr_data.get("general_info", {}),
            "open_roles": hr_data.get("open_roles", []),
            "application_instructions": hr_data.get("application_instructions", {}),
            "important_notes": hr_data.get("important_notes", []),
        }

    def process_image(self, image: Image.Image, ai_service: str, openai_model: Optional[str], gemini_model: Optional[str]) -> Optional[Dict]:
        logger.debug(f"process_image input: image size {image.size}, format {image.format}, mode {image.mode}")
        """Processes an image to extract contacts by first extracting text."""
        logger.info(f"Starting image processing with service: {ai_service}")
        logger.info(f"Image size: {image.size}, format: {image.format}, mode: {image.mode}")
        
        extracted_text = self.extract_text_from_image(image, ai_service, openai_model, gemini_model)

        if not extracted_text:
            logger.error("Failed to extract text from image.")
            st.error("Could not extract text from image. Try another AI service or check API keys.")
            return None
            
        logger.info(f"Extracted {len(extracted_text)} characters from image.")
        logger.debug(f"Extracted text preview: {extracted_text[:200]}...")
        return self.process_text(extracted_text, ai_service, openai_model, gemini_model)

    # -------------------- Structuring and Saving --------------------
    def structure_final_results(self, llm_results: list) -> List[Dict]:
        logger.debug(f"structure_final_results input: {llm_results}")
        """Formats the final list of contacts from LLM results."""
        structured = []
        for model_name, result_data in llm_results:
            contacts = result_data.get("contacts", [])
            if not isinstance(contacts, list):
                logger.warning(f"Contacts data is not a list: {contacts}")
                continue
            for contact in contacts:
                c = dict(contact)
                c["source"] = f"AI-Refined ({model_name})"
                c["confidence"] = c.get("confidence", "medium")
                c["ai_refinements"] = result_data.get("refinements", {})
                structured.append(c)
        return structured

    def save_results(self, results: Dict) -> Tuple[Path, Optional[Path]]:
        logger.debug(f"save_results input: {results}")
        """Saves extraction results to JSON, CSV, and the database."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_stem = f"contact_extraction_{timestamp}"
        
        logger.info(f"Saving results to files with stem: {filename_stem}")
        json_path = DATA_DIR / f"{filename_stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved JSON to {json_path}")
        
        csv_path = None
        if results.get("structured_contacts"):
            df = pd.DataFrame(results["structured_contacts"])
            csv_path = DATA_DIR / f"{filename_stem}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to {csv_path}")

        try:
            save_contacts_to_db(results, timestamp, filename_stem)
            logger.info("Successfully saved to database.")
            st.success("✅ Results saved to the database.")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            st.error(f"Error saving to database: {e}")
            
        return json_path, csv_path

# -------------------- UI Helper Functions --------------------
## Refactor Note: UI logic is broken into smaller functions for clarity.
def setup_sidebar():
    """Renders the Streamlit sidebar for configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("API Keys")
        st.text_input("OpenAI API Key", type="password", key=SESS_KEY_OPENAI_API)
        st.text_input("Google Gemini API Key", type="password", key=SESS_KEY_GOOGLE_API)

        st.session_state[SESS_KEY_EXTRACTOR].set_api_keys(
            st.session_state.get(SESS_KEY_OPENAI_API),
            st.session_state.get(SESS_KEY_GOOGLE_API)
        )
        
        st.success("✅ OpenAI API configured" if st.session_state.get(SESS_KEY_OPENAI_API) else "⚠️ OpenAI API not configured")
        st.success("✅ Gemini API configured" if st.session_state.get(SESS_KEY_GOOGLE_API) else "⚠️ Gemini API not configured")

        st.markdown("---")
        st.subheader("🧩 Model Selection")
        st.selectbox("OpenAI model", options=list(OPENAI_TEXT_MODELS), key=SESS_KEY_OPENAI_MODEL)
        st.selectbox("Gemini model", options=list(GEMINI_TEXT_MODELS), key=SESS_KEY_GEMINI_MODEL)
        
        st.markdown("---")
        # You can add the "Previous Extractions" loader here if needed

def handle_text_input() -> str:
    """Renders the UI for text input and returns the content."""
    return st.text_area(
        "Paste job posting, email signature, or contact information here:",
        height=300,
        placeholder="Example:\nHR - TechCorp Solutions\nSarah Johnson, Recruiter\nsarah@techcorp.com",
    )

def handle_image_input() -> Optional[Image.Image]:
    """Renders the UI for image input and returns a PIL Image object. Supports file upload and clipboard paste."""
    st.markdown("**Upload or paste an image (from clipboard):**")
    img_file = st.file_uploader("Upload a screenshot or image", type=["png", "jpg", "jpeg"])
    pasted_image = None
    paste_result = pbutton("📋 Paste an image from clipboard")
    if paste_result and paste_result.image_data is not None:
        st.info("Image data received from clipboard.")
        pasted_image = paste_result.image_data
        if pasted_image is not None and isinstance(pasted_image, Image.Image):
            st.image(pasted_image, caption="Pasted Image")
            return pasted_image
        else:
            st.error("Clipboard data is not a valid image.")
    if img_file:
        try:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image")
            return image
        except Exception as e:
            st.error(f"Error loading image: {e}")
    return None

def display_results():
    """Renders the results tabs in the UI based on session state."""
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

        open_roles = results.get("open_roles", [])
        if open_roles:
            st.subheader("Open Roles")
            st.dataframe(pd.DataFrame(open_roles))
        application_instructions = results.get("application_instructions", {})
        if application_instructions:
            st.subheader("Application Instructions")
            st.json(application_instructions)
        important_notes = results.get("important_notes", [])
        if important_notes:
            st.subheader("Important Notes")
            for note in important_notes:
                st.write(f"- {note}")

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
            from database import get_all_ai_jsons_df
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

# -------------------- Main Application Logic --------------------
def main():
    logger.debug("Starting main application loop.")
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Job Contact Extractor", page_icon="👔", layout="wide")
    st.title("👔 Job Search Contact Extractor")
    st.markdown("Extract HR/recruiter contact details from text or images using AI.")

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
                # Force a rerun to update the results panel immediately
                st.rerun()

    with col2:
        display_results()


if __name__ == "__main__":
    main()