#!/usr/bin/env python3
"""
Test script to verify the job search contact extractor installation and functionality
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('cv2', 'opencv-python'),
        ('pytesseract', 'pytesseract'),
        ('PIL', 'Pillow'),
        ('pandas', 'pandas'),
        ('streamlit', 'streamlit'),
        ('numpy', 'numpy'),
        ('easyocr', 'easyocr')
    ]
    
    optional_packages = [
        ('openai', 'openai'),
        ('google.generativeai', 'google-generativeai'),
        ('phonenumbers', 'phonenumbers'),
        ('email_validator', 'email-validator')
    ]
    
    print("\n📦 Checking required dependencies...")
    missing_required = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_required.append(package_name)
    
    print("\n📦 Checking optional dependencies...")
    missing_optional = []
    
    for import_name, package_name in optional_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"⚠️  {package_name} - OPTIONAL (for AI enhancement)")
            missing_optional.append(package_name)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("   Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n💡 Optional packages not installed: {', '.join(missing_optional)}")
        print("   For AI enhancement, install with: pip install " + " ".join(missing_optional))
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is available"""
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (200, 50), color='white')
        # We can't easily add text without additional dependencies, so just test if pytesseract runs
        
        # Try to get tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR version: {version}")
        return True
        
    except pytesseract.TesseractNotFoundError:
        print("❌ Tesseract OCR not found")
        print("   Please install Tesseract OCR:")
        print("   - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - macOS: brew install tesseract")
        print("   - Ubuntu: sudo apt install tesseract-ocr")
        return False
    except Exception as e:
        print(f"⚠️  Tesseract check failed: {str(e)}")
        return False

def check_project_files():
    """Check if all project files are present"""
    required_files = [
        'app.py',
        'batch_processor.py',
        'utils.py',
        'cli.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("\n📁 Checking project files...")
    missing_files = []
    
    for filename in required_files:
        file_path = Path(filename)
        if file_path.exists():
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - MISSING")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n❌ Missing project files: {', '.join(missing_files)}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without requiring images"""
    try:
        print("\n🧪 Testing basic functionality...")
        
        # Test ContactExtractor import
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from app import ContactExtractor
        from utils import ContactValidator, analyze_extraction_quality
        
        print("✅ Successfully imported ContactExtractor")
        
        # Test contact validation
        validator = ContactValidator()
        
        # Test name validation
        assert validator.is_valid_name("John Doe") == True
        assert validator.is_valid_name("123") == False
        assert validator.is_valid_name("") == False
        print("✅ Contact validation working")
        
        # Test quality analysis
        test_results = {
            'structured_contacts': [
                {'name': 'John Doe', 'email': 'john@example.com', 'phone': '+1-555-0123', 'title': 'HR Manager'}
            ]
        }
        
        quality = analyze_extraction_quality(test_results)
        assert quality['total_contacts'] == 1
        assert quality['contacts_with_email'] == 1
        print("✅ Quality analysis working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n🔑 Checking API configuration...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GOOGLE_API_KEY')
    
    if openai_key and openai_key != 'your_openai_api_key_here':
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  OpenAI API key not configured (optional)")
    
    if gemini_key and gemini_key != 'your_google_api_key_here':
        print("✅ Google Gemini API key configured")
    else:
        print("⚠️  Google Gemini API key not configured (optional)")
    
    if not openai_key and not gemini_key:
        print("💡 No AI API keys configured - basic OCR extraction only")
    
    return True

def run_all_tests():
    """Run all tests and return overall status"""
    print("🔍 Job Search Contact Extractor - System Check")
    print("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Tesseract OCR", check_tesseract),
        ("Project Files", check_project_files),
        ("Basic Functionality", test_basic_functionality),
        ("API Configuration", check_api_keys)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Or try: python cli.py <image_path>")
        return True
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
