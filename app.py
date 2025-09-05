import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime
import os
from pathlib import Path
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from PIL import Image
import base64
import io
import requests
from streamlit_paste_button import paste_image_button as pbutton
from typing import cast

# Load environment variables
load_dotenv()


class ContactExtractor:
    """Extracts contacts from text or images using patterns and optional LLMs."""

    def __init__(self):
        # Initialize OpenAI
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize Gemini config (create model on demand later)
        self.gemini_enabled = False
        if os.getenv("GOOGLE_API_KEY"):
            try:
                _configure = getattr(genai, "configure", None)
                if callable(_configure):
                    _configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    self.gemini_enabled = True
                else:
                    self.gemini_enabled = False
            except Exception:
                self.gemini_enabled = False

        # Storage dir
        self.data_dir = Path("extracted_contacts")
        self.data_dir.mkdir(exist_ok=True)

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

    # -------------------- Pattern extraction --------------------
    def extract_contact_patterns(self, text: str) -> dict:
        contacts = {
            "emails": [],
            "phones": [],
            "names": [],
            "titles": [],
            "companies": [],
            "linkedin": [],
            "other_social": [],
        }

        # Emails
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        for email in re.findall(email_pattern, text, re.IGNORECASE):
            try:
                validated = validate_email(email)
                contacts["emails"].append(validated.email)
            except EmailNotValidError:
                pass

        # Phones
        phone_patterns = [
            r"\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",
            r"\+?(\d{1,3})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})",
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            r"\+\d{10,15}",
        ]
        for pattern in phone_patterns:
            for phone in re.findall(pattern, text):
                try:
                    phone_str = "".join(phone) if isinstance(phone, tuple) else phone
                    parsed = phonenumbers.parse(phone_str, "US")
                    if phonenumbers.is_valid_number(parsed):
                        formatted = phonenumbers.format_number(
                            parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                        )
                        contacts["phones"].append(formatted)
                except Exception:
                    clean = re.sub(r"[^\d+]", "", "".join(phone) if isinstance(phone, tuple) else phone)
                    if len(clean) >= 10:
                        contacts["phones"].append(clean)

        # Names
        name_patterns = [
            r"(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s*(?:HR|Human Resources|Recruiter|Manager|Director|VP|CEO|CTO|Lead))",
            r"Contact:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"From:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        for pattern in name_patterns:
            contacts["names"].extend(re.findall(pattern, text, re.IGNORECASE))

        # Titles
        title_patterns = [
            r"\b(?:HR\s+)?(?:Manager|Director|Lead|Coordinator|Specialist|Representative|Recruiter|Officer)\b",
            r"\b(?:Human\s+Resources?|Talent\s+Acquisition|People\s+Operations?)\b",
            r"\b(?:CEO|CTO|VP|President|Head\s+of)\b",
        ]
        for pattern in title_patterns:
            contacts["titles"].extend(re.findall(pattern, text, re.IGNORECASE))

        # LinkedIn
        linkedin_pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9-]+"
        contacts["linkedin"].extend(re.findall(linkedin_pattern, text, re.IGNORECASE))

        # Companies
        company_patterns = [
            r"([A-Z][A-Za-z\s&]+)(?:\s+(?:Inc\.?|LLC|Corp\.?|Ltd\.?|Company|Solutions?|Technologies?))",
            r"@([A-Za-z\s&]+)\.com",
        ]
        for pattern in company_patterns:
            contacts["companies"].extend(re.findall(pattern, text))

        # Deduplicate and clean
        for k, vals in contacts.items():
            dedup = []
            seen = set()
            for v in vals:
                v2 = v.strip()
                if v2 and v2 not in seen:
                    dedup.append(v2)
                    seen.add(v2)
            contacts[k] = dedup
        return contacts

    # -------------------- LLM enhancement --------------------
    def enhance_with_llm(self, text: str, pattern_results: dict | None = None,
                         provider_preference: str | None = None,
                         openai_model: str | None = None,
                         gemini_model: str | None = None) -> list[tuple[str, dict]]:
        # Pattern summary for the prompt
        pattern_summary = ""
        if pattern_results:
            pattern_summary = f"""
        PATTERN-BASED EXTRACTION RESULTS (for reference):
        - Emails found: {', '.join(pattern_results.get('emails', [])) or 'None'}
        - Phone numbers found: {', '.join(pattern_results.get('phones', [])) or 'None'}
        - Names found: {', '.join(pattern_results.get('names', [])) or 'None'}
        - Job titles found: {', '.join(pattern_results.get('titles', [])) or 'None'}
        - Companies found: {', '.join(pattern_results.get('companies', [])) or 'None'}
        - LinkedIn profiles found: {', '.join(pattern_results.get('linkedin', [])) or 'None'}
        """

        prompt = f"""
        You are an expert at extracting and REFINING contact information from job-related text and images.
        Review pattern-based results, then refine using the original text.

        Return ONLY JSON with this structure:
        {{
            "contacts": [{{"name": "", "title": "", "company": "", "email": "", "phone": "", "linkedin": "", "department": "", "confidence": "high/medium/low", "notes": ""}}],
            "general_info": {{"company": "", "department": "", "job_posting_title": "", "location": ""}},
            "refinements": {{"pattern_accuracy": "high/medium/low", "additional_contacts_found": "0", "corrections_made": ""}}
        }}

        {pattern_summary}

        ORIGINAL TEXT TO ANALYZE:
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
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You extract accurate contact information and output strict JSON."},
                        {"role": "user", "content": prompt.format(text=text)}
                    ],
                    temperature=0.1,
                )
                content = resp.choices[0].message.content or ""
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    m = re.search(r"\{[\s\S]*\}", content)
                    data = json.loads(m.group()) if m else {}
                if data:
                    results.append((model_name, data))
            except Exception as e:
                st.warning(f"OpenAI API error: {e}")

        # Gemini
        if use_gemini:
            try:
                gm = self._select_gemini_model(gemini_model, "text")
                _GM = getattr(genai, "GenerativeModel", None)
                if _GM is None:
                    raise RuntimeError("Gemini GenerativeModel not available in this package version")
                runtime = _GM(gm)
                resp = runtime.generate_content(prompt.format(text=text))
                content = getattr(resp, "text", "") or ""
                if content:
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        m = re.search(r"\{[\s\S]*\}", content)
                        data = json.loads(m.group()) if m else {}
                    if data:
                        results.append((gm, data))
            except Exception as e:
                st.warning(f"Gemini API error: {e}")

        return results

    # -------------------- Image helpers --------------------
    def encode_image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def extract_text_from_image_openai(self, image: Image.Image, custom_prompt: str | None = None, model: str | None = None) -> str | None:
        if not self.openai_client:
            return None
        try:
            base64_image = self.encode_image_to_base64(image)
            prompt = custom_prompt or (
                "Extract all text from this image, focusing on names, titles, emails, phones, companies, LinkedIn."
            )
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
                max_tokens=2000,
            )
            return resp.choices[0].message.content or None
        except Exception as e:
            st.error(f"Error extracting text from image with OpenAI: {e}")
            return None

    def extract_text_from_image_gemini(self, image: Image.Image, custom_prompt: str | None = None, model: str | None = None) -> str | None:
        if not self.gemini_enabled:
            return None
        try:
            prompt = custom_prompt or (
                "Extract all text from this image. Focus on contact info: names, emails, phones, titles, companies."
            )
            _GM = getattr(genai, "GenerativeModel", None)
            if _GM is None:
                raise RuntimeError("Gemini GenerativeModel not available in this package version")
            runtime = _GM(self._select_gemini_model(model, "vision"))
            resp = runtime.generate_content([prompt, image])
            return getattr(resp, "text", None)
        except Exception as e:
            st.error(f"Error extracting text from image with Gemini: {e}")
            return None

    # -------------------- Pipelines --------------------
    def process_text(self, text: str, auto_ai_refinement: bool = True,
                     ai_service: str | None = None,
                     openai_model: str | None = None,
                     gemini_model: str | None = None) -> dict:
        all_results = {
            "raw_text": text,
            "contact_patterns": {},
            "llm_enhanced": [],
            "structured_contacts": [],
        }
        patterns = self.extract_contact_patterns(text)
        all_results["contact_patterns"] = patterns

        if text.strip():
            if auto_ai_refinement:
                llm = self.enhance_with_llm(text, patterns, provider_preference=ai_service, openai_model=openai_model, gemini_model=gemini_model)
            else:
                llm = self.enhance_with_llm(text, provider_preference=ai_service, openai_model=openai_model, gemini_model=gemini_model)
            all_results["llm_enhanced"] = llm

        all_results["structured_contacts"] = self.structure_final_results(all_results)
        return all_results

    def process_image(self, image: Image.Image, use_ai: bool = True, ai_service: str = "openai",
                      custom_prompt: str | None = None,
                      auto_ai_refinement: bool = True,
                      openai_model: str | None = None,
                      gemini_model: str | None = None) -> dict | None:
        extracted_text: str | None = None
        if use_ai:
            if ai_service == "openai" and self.openai_client:
                extracted_text = self.extract_text_from_image_openai(image, custom_prompt, model=openai_model)
            elif ai_service == "gemini" and self.gemini_enabled:
                extracted_text = self.extract_text_from_image_gemini(image, custom_prompt, model=gemini_model)
        if not extracted_text:
            st.error("Could not extract text from image. Try a different AI service or check your API keys.")
            return None
        return self.process_text(extracted_text, auto_ai_refinement, ai_service=ai_service, openai_model=openai_model, gemini_model=gemini_model)

    # -------------------- Structuring and saving --------------------
    def structure_final_results(self, all_results: dict) -> list[dict]:
        structured: list[dict] = []
        for model_name, llm_result in all_results.get("llm_enhanced", []):
            for contact in llm_result.get("contacts", []) or []:
                c = dict(contact)
                c["source"] = f"AI-Refined ({model_name})"
                c["confidence"] = c.get("confidence", "medium")
                c["ai_refinements"] = llm_result.get("refinements", {})
                structured.append(c)
        if structured:
            return structured

        # Fallback: build from patterns
        p = all_results.get("contact_patterns", {})
        names = p.get("names", [])
        emails = p.get("emails", [])
        phones = p.get("phones", [])
        titles = p.get("titles", [])
        companies = p.get("companies", [])
        linkedin = p.get("linkedin", [])

        for i, name in enumerate(names):
            structured.append({
                "name": name,
                "title": titles[i] if i < len(titles) else "",
                "company": companies[0] if companies else "",
                "email": emails[i] if i < len(emails) else "",
                "phone": phones[i] if i < len(phones) else "",
                "linkedin": linkedin[i] if i < len(linkedin) else "",
                "source": "Pattern Matching",
                "confidence": "low",
                "notes": "",
            })
        max_items = max(len(emails), len(phones), len(linkedin), 0)
        for i in range(max_items):
            if i >= len(names):
                structured.append({
                    "name": "",
                    "title": titles[i] if i < len(titles) else "",
                    "company": companies[0] if companies else "",
                    "email": emails[i] if i < len(emails) else "",
                    "phone": phones[i] if i < len(phones) else "",
                    "linkedin": linkedin[i] if i < len(linkedin) else "",
                    "source": "Pattern Matching",
                    "confidence": "low",
                    "notes": "Additional contact info found",
                })
        return structured

    def save_results(self, results: dict, filename: str | None = None):
        if filename is None:
            filename = f"contact_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        csv_path = None
        if results.get("structured_contacts"):
            df = pd.DataFrame(results["structured_contacts"])
            csv_path = self.data_dir / f"{filename}.csv"
            df.to_csv(csv_path, index=False)
        return json_path, csv_path


# -------------------- UI --------------------

def main():
    st.set_page_config(page_title="Job Search Contact Extractor", page_icon="👔", layout="wide")
    st.title("👔 Job Search Contact Extractor")
    st.markdown("Upload screenshots or text; extract HR/recruiter contact details.")

    if "extractor" not in st.session_state:
        st.session_state.extractor = ContactExtractor()

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("AI Keys (Optional)")
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        gemini_key = st.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        st.success("✅ OpenAI API configured" if openai_key else "⚠️ OpenAI API not configured")
        st.success("✅ Gemini API configured" if gemini_key else "⚠️ Gemini API not configured")

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
        uploaded_image: Image.Image | None = None

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
                        uploaded_image = Image.open(img_file)
                        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
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
                    if img is not None:
                        uploaded_image = img
                        st.image(uploaded_image, caption="Pasted Image", use_column_width=True)
            if img_method == "Paste from Clipboard":
                st.markdown("📋 Paste an image from your clipboard")
                paste_result = pbutton("📋 Paste an image")
                if paste_result.image_data is not None:
                    uploaded_image = cast(Image.Image, paste_result.image_data)
                    st.image(uploaded_image, caption="Pasted Image", use_column_width=True)

            st.markdown("---")
            with st.expander("🔍 Current Extraction Prompt", expanded=False):
                st.text_area("AI will use this prompt:", value=st.session_state.get("extraction_prompt", ""), height=100, disabled=True)

        st.subheader("🖼️ Image Extraction Settings")
        default_prompt = (
            "Extract all text. Focus on contact info: names, phones, emails, job titles, companies."
        )
        extraction_prompt = st.text_area(
            "Image Extraction Prompt:",
            value=st.session_state.get("extraction_prompt", default_prompt),
            height=120,
        )
        st.session_state.extraction_prompt = extraction_prompt
        preset_prompts = {
            "Default": default_prompt,
            "Job Postings": "Extract job posting text; focus HR contact info, company, department, how to apply.",
            "LinkedIn Profiles": "Extract profile text; focus name, title, company, contact links.",
            "Email Signatures": "Extract signature; focus name, title, company, email, phones, address, links.",
            "Business Cards": "Extract all fields; preserve formatting when possible.",
        }
        selected_preset = st.selectbox("Or choose a preset:", ["Custom"] + list(preset_prompts.keys()))
        if selected_preset != "Custom":
            st.session_state.extraction_prompt = preset_prompts[selected_preset]
            st.rerun()

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            use_ai_enhancement = st.checkbox("🧠 Use AI Enhancement", value=True)
        with c2:
            auto_ai_refinement = st.checkbox("🔄 Auto AI Refinement", value=True)
        ai_service = st.selectbox("AI Service:", ["openai", "gemini"]) 

        can_process = (text_content.strip() if input_method != "Upload Image/Screenshot" else uploaded_image is not None)
        if st.button("🔍 Extract Contacts", type="primary", disabled=not can_process):
            selected_openai_model = st.session_state.get("openai_model")
            selected_gemini_model = st.session_state.get("gemini_model")
            if input_method == "Upload Image/Screenshot" and uploaded_image:
                with st.spinner("Extracting text from image and processing contacts..."):
                    custom_prompt = st.session_state.get("extraction_prompt")
                    results = st.session_state.extractor.process_image(
                        uploaded_image,
                        use_ai=use_ai_enhancement,
                        ai_service=ai_service,
                        custom_prompt=custom_prompt,
                        auto_ai_refinement=auto_ai_refinement,
                        openai_model=selected_openai_model,
                        gemini_model=selected_gemini_model,
                    )
                    if results:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        json_path, csv_path = st.session_state.extractor.save_results(results, filename=f"image_extraction_{ts}")
                        st.session_state.current_results = results
                        st.session_state.current_timestamp = ts
                        st.success(f"✅ Processing complete! Results saved to {json_path.name}")
            elif text_content.strip():
                with st.spinner("Extracting contact information..."):
                    results = st.session_state.extractor.process_text(
                        text_content,
                        auto_ai_refinement,
                        ai_service=ai_service,
                        openai_model=selected_openai_model,
                        gemini_model=selected_gemini_model,
                    )
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_path, csv_path = st.session_state.extractor.save_results(results, filename=f"text_extraction_{ts}")
                    st.session_state.current_results = results
                    st.session_state.current_timestamp = ts
                    st.success(f"✅ Extraction done! Results saved to {json_path}")
                    if csv_path:
                        st.success(f"📊 CSV file saved to {csv_path}")

    with col2:
        st.header("📊 Extraction Results")
        if "current_results" in st.session_state:
            results = st.session_state.current_results
            timestamp = st.session_state.current_timestamp
            st.subheader(f"Results from: {timestamp}")
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Structured Contacts", "🔍 Pattern Analysis", "📝 Raw Text", "🤖 AI Results"])
            with tab1:
                if results.get("structured_contacts"):
                    df = pd.DataFrame(results["structured_contacts"])
                    st.dataframe(df, use_container_width=True)
                    st.download_button("📥 Download as CSV", df.to_csv(index=False), file_name=f"contacts_{timestamp}.csv", mime="text/csv")
                    st.download_button("📥 Download as JSON", json.dumps(results, indent=2, default=str), file_name=f"contacts_{timestamp}.json", mime="application/json")
                else:
                    st.info("No structured contacts found.")
            with tab2:
                patterns = results.get("contact_patterns", {})
                for category, items in patterns.items():
                    if items:
                        st.subheader(category.replace("_", " ").title())
                        for item in items:
                            st.write(f"• {item}")
            with tab3:
                st.text_area("Raw input text:", value=results.get("raw_text", ""), height=300, disabled=True)
            with tab4:
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
                st.dataframe(df, use_container_width=True)
        else:
            st.info("👆 Enter some text to start extracting contact information!")

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
