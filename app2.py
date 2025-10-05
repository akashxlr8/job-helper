import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime
import os
from pathlib import Path
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
from openai import OpenAI

from PIL import Image
import base64
import io
import requests
from streamlit_paste_button import paste_image_button as pbutton
from typing import cast
from database import save_contacts_to_db, get_all_contacts_df
from loguru import logger

# Setup loguru logging config
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "job-helper.log")
logger.remove()
logger.add(LOG_PATH, rotation="10 MB", retention="10 days", level="INFO", enqueue=True)
logger.add(lambda msg: print(msg, end=""), level="WARNING")





class ContactExtractor:
    """Extracts contacts from text or images using LLMs only."""

    def __init__(self):
        self.openai_client = None
        self.gemini_enabled = False
        # Storage dir
        self.data_dir = Path("extracted_contacts")
        self.data_dir.mkdir(exist_ok=True)

    def set_api_keys(self, openai_api_key: str | None, google_api_key: str | None):
        # Initialize OpenAI
        self.openai_client = None
        if openai_api_key:
            logger.info("Configuring OpenAI API client")
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI API client configured successfully")
        else:
            logger.warning("No OpenAI API key provided")

        # Initialize Gemini config
        self.gemini_enabled = False
        if google_api_key and GENAI_AVAILABLE:
            try:
                logger.info("Configuring Gemini API")
                _configure = getattr(genai, "configure", None)
                if callable(_configure):
                    _configure(api_key=google_api_key)
                    self.gemini_enabled = True
                    logger.info("Gemini API configured successfully")
                else:
                    self.gemini_enabled = False
                    logger.warning("Gemini configure method not callable")
            except Exception as e:
                self.gemini_enabled = False
                logger.error(f"Failed to configure Gemini API: {e}")
        elif google_api_key and not GENAI_AVAILABLE:
            logger.warning("Google API key provided but google.generativeai package not available")
        else:
            logger.warning("No Google API key provided")

        # Model defaults and capability sets
        self._default_openai_text = "gpt-4o"
        self._default_openai_vision = "gpt-4o"
        self._openai_text_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-5",  # placeholder, will fallback if unavailable
        }
        self._openai_vision_models = {"gpt-4o", "gpt-4o-mini"}

        self._default_gemini_text = "gemini-pro"
        self._default_gemini_vision = "gemini-pro-vision"
        self._gemini_text_models = {
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-pro",
        }
        self._gemini_vision_models = {
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-pro-vision",
        }

    # -------------------- Helpers --------------------
    def _select_openai_model(self, requested: str | None, purpose: str) -> str:
        if purpose == "vision":
            return requested if requested in self._openai_vision_models else self._default_openai_vision
        return requested if requested in self._openai_text_models else self._default_openai_text

    def _select_gemini_model(self, requested: str | None, purpose: str) -> str:
        if purpose == "vision":
            return requested if requested in self._gemini_vision_models else self._default_gemini_vision
        return requested if requested in self._gemini_text_models else self._default_gemini_text


    def _create_fallback_json_from_text(self, ai_response: str, original_text: str) -> dict | None:
        """Create a basic JSON structure when AI response can't be parsed"""
        try:
            # Try to extract basic contact info from the AI response text
            fallback_contacts = []
            
            # Look for common patterns in the AI response
            name_match = re.search(r"(?:Name|name):\s*([^\n\r]+)", ai_response)
            title_match = re.search(r"(?:Job Title|Title|title|Position):\s*([^\n\r]+)", ai_response)
            company_match = re.search(r"(?:Company|company|Organization):\s*([^\n\r]+)", ai_response)
            email_match = re.search(r"(?:Email|email):\s*([^\s\n\r]+@[^\s\n\r]+)", ai_response)
            
            if name_match or email_match or title_match or company_match:
                contact = {
                    "name": name_match.group(1).strip() if name_match else "",
                    "title": title_match.group(1).strip() if title_match else "",
                    "company": company_match.group(1).strip() if company_match else "",
                    "email": email_match.group(1).strip() if email_match else "",
                    "phone": "",
                    "linkedin": "",
                    "department": "",
                    "confidence": "medium",
                    "notes": "Extracted from AI response text (fallback parsing)"
                }
                # Only add if we have at least one meaningful field
                if any([contact["name"], contact["email"], contact["title"], contact["company"]]):
                    fallback_contacts.append(contact)
            
            if fallback_contacts:
                return {
                    "contacts": fallback_contacts,
                    "general_info": {
                        "company": fallback_contacts[0]["company"] if fallback_contacts else "",
                        "department": "",
                        "job_posting_title": "",
                        "location": ""
                    },
                    "refinements": {
                        "additional_contacts_found": str(len(fallback_contacts)),
                        "corrections_made": "Fallback parsing due to AI JSON parse error"
                    }
                }
            return None
        except Exception as e:
            logger.error(f"Fallback JSON creation failed: {e}")
            return None

    # -------------------- LLM enhancement --------------------
    def enhance_with_llm(self, text: str, provider_preference: str | None = None,
                         openai_model: str | None = None,
                         gemini_model: str | None = None) -> list[tuple[str, dict]]:
        prompt = f"""You are an expert at extracting contact information from job-related text and images.

        Extract contact information and return it as a JSON object with this exact structure:
        {{
            "contacts": [{{"name": "", "title": "", "company": "", "email": "", "phone": "", "linkedin": "", "department": "", "confidence": "high/medium/low", "notes": ""}}],
            "general_info": {{"company": "", "department": "", "job_posting_title": "", "location": ""}},
            "refinements": {{"additional_contacts_found": "0", "corrections_made": ""}}
        }}

        TEXT TO ANALYZE:
        {{text}}
        """

        results: list[tuple[str, dict]] = []
        use_openai = self.openai_client is not None and (provider_preference in (None, "openai"))
        use_gemini = self.gemini_enabled and (provider_preference in (None, "gemini"))

        # OpenAI
        if use_openai:
            try:
                model_name = self._select_openai_model(openai_model, "text")
                client = self.openai_client
                if client is None:
                    raise RuntimeError("OpenAI client not initialized")
                
                formatted_prompt = prompt.format(text=text)
                logger.info(f"OpenAI request: model={model_name}, prompt length={len(formatted_prompt)}")
                logger.debug(f"OpenAI formatted prompt: {formatted_prompt}")
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=8000,
                    temperature=0.1,
                )
                content = resp.choices[0].message.content or ""
                logger.info(f"OpenAI response: {content}")
                
                # Clean up the content - remove code blocks, instructions, and strip whitespace
                cleaned_content = content.strip()
                
                # Remove any instruction text that might have been included
                instruction_patterns = [
                    r"You are an expert at extracting contact information",
                    r"Extract contact information and return it as a JSON object",
                    r"TEXT TO ANALYZE",
                ]
                for pattern in instruction_patterns:
                    cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE)
                
                # Remove markdown code blocks
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content[3:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()
                
                # Remove any remaining instruction-like text
                cleaned_content = re.sub(r"^\s*[\w\s,.:;-]*?(?=\{)", "", cleaned_content, flags=re.MULTILINE)
                cleaned_content = cleaned_content.strip()
                
                try:
                    data = json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    logger.error(f"OpenAI JSON parse error: {e}")
                    logger.error(f"OpenAI response content: {repr(content)}")
                    
                    # Try to extract JSON using regex
                    m = re.search(r"\{[\s\S]*\}", cleaned_content)
                    if m:
                        try:
                            potential_json = m.group().strip()
                            data = json.loads(potential_json)
                            logger.info(f"Successfully parsed JSON using regex fallback")
                        except Exception as e2:
                            logger.error(f"OpenAI fallback JSON parse failed: {e2}")
                            logger.error(f"Attempted JSON: {repr(potential_json if 'm' in locals() else 'None')}")
                            
                            # Create basic fallback JSON from the original response
                            data = self._create_fallback_json_from_text(content, text)
                            if data:
                                logger.info(f"Created fallback JSON structure from response text")
                            else:
                                st.warning(f"OpenAI API error: Could not parse response as JSON. See logs for details.")
                    else:
                        logger.error(f"No JSON structure found in response")
                        
                        # Create basic fallback JSON from the original response  
                        data = self._create_fallback_json_from_text(content, text)
                        if data:
                            logger.info(f"Created fallback JSON structure from response text")
                        else:
                            st.warning(f"OpenAI API error: Could not find JSON in response. See logs for details.")
                if data:
                    results.append((model_name, data))
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                st.warning(f"OpenAI API error: {e}")

        # Gemini
        if use_gemini:
            try:
                gm = self._select_gemini_model(gemini_model, "text")
                logger.info(f"Gemini request: model={gm}, prompt={prompt}")
                try:
                    _GM = getattr(genai, "GenerativeModel", None) if GENAI_AVAILABLE else None
                    if _GM is None:
                        raise RuntimeError("Gemini GenerativeModel not available in this package version")
                    runtime = _GM(gm)
                    resp = runtime.generate_content(prompt.format(text=text))
                    content = getattr(resp, "text", "") or ""
                    logger.info(f"Gemini response: {content}")
                    if content:
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            m = re.search(r"\{[\s\S]*\}", content)
                            data = json.loads(m.group()) if m else {}
                        if data:
                            results.append((gm, data))
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
                    st.warning(f"Gemini API error: {e}")
            except Exception as e:
                st.warning(f"Gemini API error: {e}")

        return results

    # -------------------- Image helpers --------------------
    def encode_image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def extract_text_from_image_openai(self, image: Image.Image, model: str | None = None) -> str | None:
        if not self.openai_client:
            return None
        try:
            model_name = self._select_openai_model(model, "vision")
            prompt = "Extract all text. Focus on contact info: names, phones, emails, job titles, companies."
            logger.info(f"OpenAI Vision request: model={model}, prompt={prompt}")
            try:
                base64_image = self.encode_image_to_base64(image)
                resp = self.openai_client.chat.completions.create(
                    model=self._select_openai_model(model, "vision"),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    max_tokens=8000,
                )
                logger.info(f"OpenAI Vision response: {resp}")
                return resp.choices[0].message.content or None
            except Exception as e:
                logger.error(f"Error extracting text from image with OpenAI: {e}")
                st.error(f"Error extracting text from image with OpenAI: {e}")
                return None
        except Exception as e:
            st.error(f"Error extracting text from image with OpenAI: {e}")
            return None

    def extract_text_from_image_gemini(self, image: Image.Image, model: str | None = None) -> str | None:
        if not self.gemini_enabled:
            return None
        try:
            logger.info(f"Gemini Vision request: model={model}")
            try:
                _GM = getattr(genai, "GenerativeModel", None) if GENAI_AVAILABLE else None
                if _GM is None:
                    raise RuntimeError("Gemini GenerativeModel not available in this package version")
                runtime = _GM(self._select_gemini_model(model, "vision"))
                prompt = "Extract all text. Focus on contact info: names, phones, emails, job titles, companies."
                resp = runtime.generate_content([prompt, image])
                logger.info(f"Gemini Vision response: {resp}")
                return getattr(resp, "text", None)
            except Exception as e:
                logger.error(f"Error extracting text from image with Gemini: {e}")
                st.error(f"Error extracting text from image with Gemini: {e}")
                return None
        except Exception as e:
            st.error(f"Error extracting text from image with Gemini: {e}")
            return None

    def _is_structured_contact_text(self, text: str) -> bool:
        """Check if text appears to be structured contact information rather than raw text"""
        logger.debug(f"Checking structured text detection on: {text[:300]}...")
        
        # Look for common structured contact patterns from image extraction
        structured_indicators = [
            r"\*\*Contact Information:\*\*",  # **Contact Information:**
            r"- \*\*Name:\*\*",  # - **Name:**
            r"- \*\*Job Title:\*\*",  # - **Job Title:**
            r"- \*\*Company:\*\*",  # - **Company:**
            r"- \*\*Email:\*\*",  # - **Email:**
            r"\*\*Name:\*\*",  # **Name:**
            r"\*\*Job Title:\*\*",  # **Job Title:**
            r"\*\*Company:\*\*",  # **Company:**
            r"\*\*Email:\*\*",  # **Email:**
            r"Name:\s*[A-Za-z]",  # Name: followed by letters
            r"Job Title:\s*[A-Za-z]",  # Job Title: followed by letters
            r"Company:\s*[A-Za-z]",  # Company: followed by letters
            r"Email:\s*[a-zA-Z0-9]",  # Email: followed by alphanumeric
        ]
        
        # Count how many structured indicators are present
        indicator_count = 0
        for pattern in structured_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                indicator_count += 1
                logger.debug(f"Pattern matched: {pattern}")
        
        # Also check if it looks like a contact card format
        contact_card_patterns = [
            r"^[A-Za-z\s]+\n[A-Za-z\s&\-\.]+\n[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Name\nTitle\nEmail
            r"Contact.*:\s*[A-Za-z]",  # "Contact: Name"
        ]
        
        card_count = sum(1 for pattern in contact_card_patterns if re.search(pattern, text, re.MULTILINE | re.IGNORECASE))
        
        logger.debug(f"Structured indicators found: {indicator_count}, card patterns: {card_count}")
        
        # If we have multiple structured indicators or contact card patterns, it's likely structured
        return indicator_count >= 2 or card_count >= 1

    def _parse_structured_contact_text(self, text: str) -> list[dict]:
        """Parse structured contact text into contact dictionaries"""
        contacts = []
        
        # Extract fields using regex
        name_match = re.search(r"(?:\*\*)?Name:(?:\*\*)?\s*([^\n\r]+)", text, re.IGNORECASE)
        title_match = re.search(r"(?:\*\*)?Job Title:(?:\*\*)?\s*([^\n\r]+)", text, re.IGNORECASE)
        company_match = re.search(r"(?:\*\*)?Company:(?:\*\*)?\s*([^\n\r]+)", text, re.IGNORECASE)
        email_match = re.search(r"(?:\*\*)?Email:(?:\*\*)?\s*([^\n\r]+)", text, re.IGNORECASE)
        
        if name_match or email_match or title_match or company_match:
            contact = {
                "name": name_match.group(1).strip() if name_match else "",
                "title": title_match.group(1).strip() if title_match else "",
                "company": company_match.group(1).strip() if company_match else "",
                "email": email_match.group(1).strip() if email_match else "",
                "phone": "",
                "linkedin": "",
                "department": "",
                "confidence": "high",
                "notes": "Direct parsing from image extraction",
                "source": "AI Image Extraction"
            }
            # Only add if we have at least one meaningful field
            if any([contact["name"], contact["email"], contact["title"], contact["company"]]):
                contacts.append(contact)
        
        return contacts

    # -------------------- Pipelines --------------------
    def process_text(self, text: str, mode: str = "ai", ai_service: str = "openai", openai_model: str | None = None, gemini_model: str | None = None) -> dict:
        """Process raw text and return structured extraction results.

        Args:
            text: input text to process
            mode: "ai" for AI-only extraction
            ai_service/openai_model/gemini_model: AI service selection
        """
        logger.info(f"Starting text processing with mode={mode}, ai_service={ai_service}")
        all_results = {
            "raw_text": text,
            "llm_enhanced": [],
            "structured_contacts": [],
            "extraction_mode": mode,
        }

        if mode == "ai":
            # AI-Only mode: skip pattern matching, use only LLM
            logger.info("AI-only mode: calling LLM for extraction")
            if text.strip():
                llm = self.enhance_with_llm(text, provider_preference=ai_service, openai_model=openai_model, gemini_model=gemini_model)
                all_results["llm_enhanced"] = llm
                logger.info(f"LLM returned {len(llm)} result(s)")

        all_results["structured_contacts"] = self.structure_final_results(all_results)
        logger.info(f"Structured {len(all_results['structured_contacts'])} contact(s)")
        return all_results

    def process_image(self, image: Image.Image, mode: str = "ai", ai_service: str = "openai",
                      openai_model: str | None = None,
                      gemini_model: str | None = None) -> dict | None:
        """Extract contacts from an image.
        
        Args:
            image: PIL Image to process
            mode: "ai" for AI-only extraction
            ai_service: which AI service to use for image text extraction
            custom_prompt: optional custom prompt for image extraction
            openai_model/gemini_model: model selection
        """
        logger.info(f"Starting image processing with mode={mode}, ai_service={ai_service}")
        extracted_text: str | None = None
        
        # Always use AI for image text extraction (OCR alternative)
        if ai_service == "openai" and self.openai_client:
            logger.info("Using OpenAI for image text extraction")
            extracted_text = self.extract_text_from_image_openai(image, model=openai_model)
        elif ai_service == "gemini" and self.gemini_enabled:
            logger.info("Using Gemini for image text extraction")
            extracted_text = self.extract_text_from_image_gemini(image, model=gemini_model)
        
        if not extracted_text:
            logger.error("Failed to extract text from image")
            st.error("Could not extract text from image. Try a different AI service or check your API keys.")
            return None
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from image")
        
        # Check if the extracted text already contains structured contact information
        is_structured = self._is_structured_contact_text(extracted_text)
        logger.info(f"Checking if extracted text is structured: {is_structured}")
        logger.debug(f"Extracted text preview: {extracted_text[:200]}...")
        if is_structured:
            logger.info("Extracted text appears to be structured contact information, parsing directly")
            structured_contacts = self._parse_structured_contact_text(extracted_text)
            if structured_contacts:
                all_results = {
                    "raw_text": extracted_text,
                    "llm_enhanced": [("direct_parse", {"contacts": structured_contacts, "general_info": {}, "refinements": {"additional_contacts_found": str(len(structured_contacts)), "corrections_made": "Direct parsing from image extraction"}})],
                    "structured_contacts": structured_contacts,
                    "extraction_mode": mode,
                }
                return all_results
        
        return self.process_text(extracted_text, mode=mode, ai_service=ai_service, openai_model=openai_model, gemini_model=gemini_model)

    # -------------------- Structuring and saving --------------------
    def structure_final_results(self, all_results: dict) -> list[dict]:
        structured: list[dict] = []
        for model_name, llm_result in all_results.get("llm_enhanced", []):
            contacts = llm_result.get("contacts")
            if contacts is None:
                st.warning(f"OpenAI API error: Missing 'contacts' key in response: {llm_result}")
                continue
            for contact in contacts or []:
                c = dict(contact)
                c["source"] = f"AI-Refined ({model_name})"
                c["confidence"] = c.get("confidence", "medium")
                c["ai_refinements"] = llm_result.get("refinements", {})
                structured.append(c)
        return structured
        return structured

    def save_results(self, results: dict, filename: str | None = None):
        if filename is None:
            filename = f"contact_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Saving results to {filename}")
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved JSON to {json_path}")
        
        csv_path = None
        if results.get("structured_contacts"):
            df = pd.DataFrame(results["structured_contacts"])
            csv_path = self.data_dir / f"{filename}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to {csv_path}")
        
        # Save to database
        try:
            save_contacts_to_db(results, datetime.now().strftime('%Y%m%d_%H%M%S'), filename)
            logger.info("Successfully saved to database")
            st.success("✅ Results saved to the database.")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            st.error(f"Error saving to database: {e}")

        return json_path, csv_path


# -------------------- UI --------------------

def main():
    st.set_page_config(page_title="Job Search Contact Extractor", page_icon="👔", layout="wide")
    st.title("👔 Job Search Contact Extractor")
    st.markdown("Upload screenshots or text; extract HR/recruiter contact details.")
    
    logger.info("App started/refreshed")

    if "extractor" not in st.session_state:
        logger.info("Initializing ContactExtractor")
        st.session_state.extractor = ContactExtractor()
    
    # Clear any cached image data to prevent media file errors
    if "uploaded_image" in st.session_state:
        del st.session_state.uploaded_image
        logger.debug("Cleared cached uploaded_image from session")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("AI Keys (Optional)")
        st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        st.text_input("Google Gemini API Key", type="password", key="google_api_key")

        if "extractor" in st.session_state:
            st.session_state.extractor.set_api_keys(st.session_state.get("openai_api_key"), st.session_state.get("google_api_key"))

        st.success("✅ OpenAI API configured" if st.session_state.get("openai_api_key") else "⚠️ OpenAI API not configured")
        st.success("✅ Gemini API configured" if st.session_state.get("google_api_key") else "⚠️ Gemini API not configured")

        st.markdown("---")
        st.subheader("🧩 Model Selection")
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5"]
        gemini_models = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-pro", "gemini-pro-vision"]
        st.session_state.openai_model = st.selectbox(
            "OpenAI model", options=openai_models,
            index=openai_models.index(st.session_state.get("openai_model", "gpt-4o")) if st.session_state.get("openai_model") in openai_models else 0,
            help="Models like gpt-4o, gpt-4.1, or gpt-4o-mini. Vision tasks auto-fallback if needed."
        )
        st.session_state.gemini_model = st.selectbox(
            "Gemini model", options=gemini_models,
            index=gemini_models.index(st.session_state.get("gemini_model", "gemini-1.5-flash")) if st.session_state.get("gemini_model") in gemini_models else 0,
            help="Use 1.5/vision variants for images."
        )

        st.markdown("---")
        st.subheader(" Previous Extractions")
        data_dir = Path("extracted_contacts")
        if data_dir.exists():
            json_files = list(data_dir.glob("*.json"))
            if json_files:
                sel = st.selectbox("Load previous extraction:", [""] + [f.stem for f in json_files])
                if sel:
                    with open(data_dir / f"{sel}.json", "r", encoding="utf-8") as f:
                        st.session_state.previous_results = json.load(f)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input Text")
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload Text File", "Upload Image/Screenshot"])
        text_content = ""
        uploaded_image = None

        if input_method == "Type/Paste Text":
            text_content = st.text_area(
                "Paste job posting, email signature, or contact information:",
                height=300,
                placeholder=(
                    "Paste here...\n\nExample:\nHR - TechCorp Solutions\nSarah Johnson, Senior Recruiter\nEmail: sarah@techcorp.com\nPhone: (555) 123-4567"
                ),
            )
        elif input_method == "Upload Text File":
            up = st.file_uploader("Choose a text file", type=["txt", "md", "csv"]) 
            if up:
                try:
                    text_content = up.read().decode("utf-8", errors="ignore")
                    st.text_area("File content preview:", value=(text_content[:500] + "..." if len(text_content) > 500 else text_content), height=200, disabled=True)
                except Exception:
                    st.error("Could not read the file. Ensure it's valid text.")
        elif input_method == "Upload Image/Screenshot":
            st.markdown("📷 Upload a screenshot or image containing job info")
            img_method = st.radio("Image input method:", ["Browse File", "Paste (URL/Base64)", "Paste from Clipboard"], horizontal=True)
            if img_method == "Browse File":
                img_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp", "tiff"]) 
                if img_file:
                    try:
                        img = Image.open(img_file)
                        if img is not None and isinstance(img, Image.Image):
                            uploaded_image = img
                            st.image(uploaded_image, caption="Uploaded Image")
                        else:
                            st.error("Uploaded file is not a valid image.")
                            uploaded_image = None
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                        uploaded_image = None
            else:
                pasted_value = st.text_area(
                    "Paste image URL or Base64 data URI (Ctrl+V)",
                    height=140,
                    placeholder=(
                        "Examples:\n- https://example.com/image.jpg\n- data:image/png;base64,iVBOR...\n- Raw base64 without prefix"
                    ),
                )
                if pasted_value:
                    img = _load_image_from_paste(pasted_value)
                    if img is not None and isinstance(img, Image.Image):
                        uploaded_image = img
                        try:
                            st.image(uploaded_image, caption="Pasted Image")
                        except Exception as e:
                            st.error(f"Error displaying pasted image: {e}")
                            uploaded_image = None
                    else:
                        st.error("Pasted data is not a valid image.")
                        uploaded_image = None
            if img_method == "Paste from Clipboard":
                st.markdown("📋 Paste an image from your clipboard")
                paste_result = pbutton("📋 Paste an image")
                if paste_result.image_data is not None:
                    st.info("Image data received from clipboard.")
                    img = paste_result.image_data
                    if img is not None and isinstance(img, Image.Image):
                        uploaded_image = img
                        try:
                            st.image(uploaded_image, caption="Pasted Image")
                        except Exception as e:
                            st.error(f"Error displaying clipboard image: {e}")
                            uploaded_image = None
                    else:
                        st.error("Clipboard data is not a valid image.")
                        uploaded_image = None
                else:
                    st.warning("No image data received from clipboard. Please try again.")

            st.markdown("---")
            with st.expander("🔍 Current Extraction Prompt", expanded=False):
                default_prompt = "Extract all text. Focus on contact info: names, phones, emails, job titles, companies."
                st.text_area("AI will use this prompt:", value=default_prompt, height=100, disabled=True)

        st.markdown("---")
        st.subheader("🎯 Extraction Mode")
        mode = "ai"
        logger.info(f"User selected extraction mode: AI-Only (mode={mode})")
        
        ai_service = st.selectbox("AI Service:", ["openai", "gemini"], help="AI service for image text extraction and AI-Only mode") 

        can_process = (text_content.strip() if input_method != "Upload Image/Screenshot" else uploaded_image is not None)
        if st.button("🔍 Extract Contacts", type="primary", disabled=not can_process):
            logger.info(f"Extract button clicked with mode={mode}, input_method={input_method}")
            selected_openai_model = st.session_state.get("openai_model")
            selected_gemini_model = st.session_state.get("gemini_model")
            if input_method == "Upload Image/Screenshot" and uploaded_image:
                logger.info("Processing image upload")
                with st.spinner("Extracting text from image and processing contacts..."):
                    results = st.session_state.extractor.process_image(
                        uploaded_image,
                        mode=mode,
                        ai_service=ai_service,
                        openai_model=selected_openai_model,
                        gemini_model=selected_gemini_model,
                    )
                    if results:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        json_path, csv_path = st.session_state.extractor.save_results(results, filename=f"image_extraction_{ts}")
                        st.session_state.current_results = results
                        st.session_state.current_timestamp = ts
                        logger.info(f"Image processing complete, saved to {json_path.name}")
                        st.success(f"✅ Processing complete! Results saved to {json_path.name}")
            elif text_content.strip():
                logger.info("Processing text input")
                with st.spinner("Extracting contact information..."):
                    results = st.session_state.extractor.process_text(
                        text_content,
                        mode=mode,
                        ai_service=ai_service,
                        openai_model=selected_openai_model,
                        gemini_model=selected_gemini_model,
                    )
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_path, csv_path = st.session_state.extractor.save_results(results, filename=f"text_extraction_{ts}")
                    st.session_state.current_results = results
                    st.session_state.current_timestamp = ts
                    logger.info(f"Text processing complete, saved to {json_path}")
                    st.success(f"✅ Extraction done! Results saved to {json_path}")
                    if csv_path:
                        st.success(f"📊 CSV file saved to {csv_path}")

    with col2:
        st.header("📊 Extraction Results")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Structured Contacts", "📝 Raw Text", "🤖 AI Results", "🗃️ Database"])

        if "current_results" in st.session_state:
            results = st.session_state.current_results
            timestamp = st.session_state.current_timestamp
            extraction_mode = results.get("extraction_mode", "unknown")
            mode_badge = "🤖 AI-Only"
            st.subheader(f"Results from: {timestamp}")
            st.info(f"Extraction Mode: {mode_badge}")
            with tab1:
                if results.get("structured_contacts"):
                    df = pd.DataFrame(results["structured_contacts"])
                    st.dataframe(df, width="stretch")
                    st.download_button("📥 Download as CSV", df.to_csv(index=False), file_name=f"contacts_{timestamp}.csv", mime="text/csv")
                    st.download_button("📥 Download as JSON", json.dumps(results, indent=2, default=str), file_name=f"contacts_{timestamp}.json", mime="application/json")
                else:
                    st.info("No structured contacts found.")
            with tab2:
                st.text_area("Raw input text:", value=results.get("raw_text", ""), height=300, disabled=True)
            with tab3:
                llm_results = results.get("llm_enhanced", [])
                if llm_results:
                    for model_name, result in llm_results:
                        st.subheader(f"{model_name} Analysis")
                        st.json(result)
                else:
                    st.info("No AI enhancement available. Add API keys to enable it.")
        elif "previous_results" in st.session_state:
            st.subheader("📂 Previous Results")
            prev = st.session_state.previous_results
            if prev.get("structured_contacts"):
                df = pd.DataFrame(prev["structured_contacts"])
                st.dataframe(df, width="stretch")
        else:
            st.info("👆 Enter some text to start extracting contact information!")

        with tab4:
            st.header("🗃️ All Contacts in Database")
            try:
                all_contacts_df = get_all_contacts_df()
                if not all_contacts_df.empty:
                    st.dataframe(all_contacts_df, width="stretch")
                    st.download_button("📥 Download All as CSV", all_contacts_df.to_csv(index=False), file_name="all_contacts.csv", mime="text/csv")
                else:
                    st.info("No contacts found in the database.")
            except Exception as e:
                st.error(f"Error loading contacts from database: {e}")

    st.markdown("---")


def _load_image_from_paste(pasted: str) -> Image.Image | None:
    try:
        pasted = pasted.strip()
        if not pasted:
            return None
        if pasted.startswith("data:image"):
            header, b64 = pasted.split(",", 1)
            img_bytes = base64.b64decode(b64)
            return Image.open(io.BytesIO(img_bytes))
        if pasted.startswith("http://") or pasted.startswith("https://"):
            resp = requests.get(pasted, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        # raw base64
        img_bytes = base64.b64decode(pasted, validate=True)
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        return None


if __name__ == "__main__":
    main()
