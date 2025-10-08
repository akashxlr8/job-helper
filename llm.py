"""LLM helpers and ContactExtractor class.

This module contains all AI/service-specific logic (OpenAI/Gemini interactions,
prompt templates, model selection, and the ContactExtractor class) so the main
app can remain focused on UI and orchestration.
"""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from loguru import logger
from PIL import Image

# Conditional Third-Party Imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

from database import save_contacts_to_db

# --- Configuration Constants used by LLM logic ---
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
        "email": "mohammed.imranullah@jiostar.com",
        "hiring_for_role": "Sr Staff Data Scientist, Staff/Sr Staff Backend Engineer",
        "yoe": "7+ yrs",
        "tech_stack": "Engineering, Security, Product, Design, Analytics",
        "location": ""
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

For each contact, extract and include these fields if present: name, title, company, email, hiring_for_role (the position(s) they are hiring for), yoe (years of experience required), tech_stack (technologies mentioned), location (job location).

Now, analyze the following text and provide the JSON output in the exact same format as the example. If a section or field is not present, use an empty string "" or an empty list [].

TEXT TO ANALYZE:
{text}
"""
VISION_PROMPT = "Extract all text. try to maintain the formatting as much as possible. Do not add any commentary or verbosity. Always include all the important text like names, emails, companies, roles, experience requirements, and application instructions. Extract the text as accurately as possible, without adding or omitting any details. Try to keep the poster details as intact as possible, if present. If it is a linked post, try to extract the name of the person posting the job, their role, company, and any contact details mentioned."

# Model Configurations
OPENAI_TEXT_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5"}
OPENAI_VISION_MODELS = {"gpt-4o", "gpt-4o-mini"}
DEFAULT_OPENAI_TEXT_MODEL = "gpt-4o"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4o"

GEMINI_TEXT_MODELS = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"}
GEMINI_VISION_MODELS = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"}
DEFAULT_GEMINI_TEXT_MODEL = "gemini-pro"
DEFAULT_GEMINI_VISION_MODEL = "gemini-pro-vision"


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
