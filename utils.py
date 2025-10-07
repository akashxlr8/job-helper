try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
from PIL import Image
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None
import re

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    @staticmethod
    def enhance_contrast(image):
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    @staticmethod
    def remove_noise(image):
        """Remove noise using morphological operations"""
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closing
    
    @staticmethod
    def sharpen_image(image):
        """Sharpen the image for better text recognition"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @staticmethod
    def deskew_image(image):
        """Correct skewed text in images"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find all white pixels
        coords = np.column_stack(np.where(gray > 0))
        
        # Find minimum area rectangle
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Correct the angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        
        return image
    
    @staticmethod
    def resize_for_ocr(image, target_height=800):
        """Resize image to optimal size for OCR"""
        height, width = image.shape[:2]
        
        if height < target_height:
            # Upscale small images
            scale_factor = target_height / height
            new_width = int(width * scale_factor)
            resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        else:
            # Downscale large images
            scale_factor = target_height / height
            new_width = int(width * scale_factor)
            resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized
    
    @classmethod
    def preprocess_pipeline(cls, image):
        """Complete preprocessing pipeline"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply preprocessing steps
        enhanced = cls.enhance_contrast(image)
        deskewed = cls.deskew_image(enhanced)
        resized = cls.resize_for_ocr(deskewed)
        sharpened = cls.sharpen_image(resized)
        denoised = cls.remove_noise(sharpened)
        
        return {
            'original': image,
            'enhanced_contrast': enhanced,
            'deskewed': deskewed,
            'resized': resized,
            'sharpened': sharpened,
            'final': denoised
        }

class ContactValidator:
    """Validate and clean extracted contact information"""
    
    @staticmethod
    def is_valid_name(name):
        """Check if extracted text looks like a valid name"""
        if not name or len(name) < 2:
            return False
        
        # Check for common patterns
        name = name.strip()
        
        # Should contain letters
        if not any(c.isalpha() for c in name):
            return False
        
        # Should not be too long
        if len(name) > 50:
            return False
        
        # Should not contain too many numbers
        if sum(c.isdigit() for c in name) > len(name) * 0.3:
            return False
        
        # Common invalid patterns
        invalid_patterns = [
            r'^\d+$',  # Only numbers
            r'^[A-Z]{3,}$',  # All caps (likely abbreviation)
            r'www\.|http|\.com|\.org',  # URLs
            r'@',  # Email parts
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def clean_company_name(company):
        """Clean and standardize company names"""
        if not company:
            return ""
        
        company = company.strip()
        
        # Remove common prefixes/suffixes
        suffixes = ['Inc.', 'Inc', 'LLC', 'Corp.', 'Corp', 'Ltd.', 'Ltd', 'Company', 'Co.']
        for suffix in suffixes:
            if company.endswith(suffix):
                company = company[:-len(suffix)].strip()
        
        return company
    
    @staticmethod
    def standardize_title(title):
        """Standardize job titles"""
        if not title:
            return ""
        
        title = title.strip()
        
        # Common mappings
        title_mappings = {
            'hr': 'Human Resources',
            'talent acquisition': 'Talent Acquisition',
            'people ops': 'People Operations',
            'mgr': 'Manager',
            'dir': 'Director',
            'vp': 'Vice President',
            'ceo': 'Chief Executive Officer',
            'cto': 'Chief Technology Officer',
            'cfo': 'Chief Financial Officer',
        }
        
        title_lower = title.lower()
        for abbrev, full in title_mappings.items():
            if abbrev in title_lower:
                title = title_lower.replace(abbrev, full).title()
        
        return title

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
