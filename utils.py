from PIL import Image

import re

def analyze_extraction_quality(results):
    """Analyze the quality of extraction results"""
    quality_metrics = {
        'total_contacts': len(results.get('structured_contacts', [])),
        'contacts_with_email': 0,
        'contacts_with_phone': 0,
        'contacts_with_name': 0,
        'contacts_with_title': 0,
        'ocr_methods_used': len(results.get('raw_texts', [])),
        'llm_enhanced': len(results.get('llm_enhanced', [])) > 0,
        'quality_score': 0
    }
    
    contacts = results.get('structured_contacts', [])
    
    for contact in contacts:
        if contact.get('email'):
            quality_metrics['contacts_with_email'] += 1
        if contact.get('phone'):
            quality_metrics['contacts_with_phone'] += 1
        if contact.get('name'):
            quality_metrics['contacts_with_name'] += 1
        if contact.get('title'):
            quality_metrics['contacts_with_title'] += 1
    
    # Calculate quality score (0-100)
    if quality_metrics['total_contacts'] > 0:
        completeness = (
            quality_metrics['contacts_with_email'] +
            quality_metrics['contacts_with_phone'] +
            quality_metrics['contacts_with_name'] +
            quality_metrics['contacts_with_title']
        ) / (quality_metrics['total_contacts'] * 4) * 100
        
        quality_metrics['quality_score'] = min(100, completeness)
    
    return quality_metrics


# ------------------ Simple passlock helpers ------------------
import hashlib
import os
from pathlib import Path


def _read_streamlit_secrets_toml():
    """Very small parser to extract `passlock` from .streamlit/secrets.toml if present.

    This mirrors how Streamlit stores secrets while avoiding importing streamlit here.
    Returns the raw string value or None.
    """
    path = Path(__file__).parent / '.streamlit' / 'secrets.toml'
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('passlock') and '=' in s:
                    k, v = s.split('=', 1)
                    v = v.strip().strip('"').strip("'")
                    return v
    except Exception:
        return None
    return None


def get_passlock_raw():
    """Return the stored passlock (raw string) from environment or Streamlit secrets.

    Priority:
    1. ENV var PASSLOCK
    2. .streamlit/secrets.toml (passlock)
    3. None if not found
    """
    v = os.environ.get('PASSLOCK')
    if v:
        return v
    v = _read_streamlit_secrets_toml()
    return v


def verify_passlock(candidate: str) -> bool:
    """Verify a candidate passlock against stored passlock.

    Supported stored formats:
    - plain string (direct compare)
    - sha256$<hex> (compare SHA256 hex digest)

    Returns True on match.
    """
    stored = get_passlock_raw()
    if not stored:
        return False
    if stored.startswith('sha256$'):
        expected = stored.split('$', 1)[1]
        h = hashlib.sha256(candidate.encode('utf-8')).hexdigest()
        return h == expected
    # fallback: plaintext compare
    return candidate == stored


def make_sha256_hash(plaintext: str) -> str:
    """Return a small helper value to put in secrets.toml: sha256$<hex>"""
    return 'sha256$' + hashlib.sha256(plaintext.encode('utf-8')).hexdigest()
