#!/bin/bash

echo "==============================================="
echo "Job Search Contact Extractor - Setup Script"
echo "==============================================="
echo

echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

python3 --version
echo

echo "[2/4] Installing Python dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi
echo

echo "[3/4] Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract OCR found: $(tesseract --version | head -1)"
else
    echo "⚠️  Tesseract OCR not found"
    echo "Install instructions:"
    echo "  macOS: brew install tesseract"
    echo "  Ubuntu/Debian: sudo apt install tesseract-ocr"
    echo "  CentOS/RHEL: sudo yum install tesseract"
fi
echo

echo "[4/4] Testing installation..."
python3 test_setup.py
if [ $? -ne 0 ]; then
    echo "SETUP FAILED: Please check the errors above"
    exit 1
fi

echo
echo "==============================================="
echo "SETUP COMPLETE!"
echo "==============================================="
echo
echo "To start the application:"
echo "  streamlit run app.py"
echo
echo "To test with a single image:"
echo "  python3 cli.py your_screenshot.png"
echo
echo "For batch processing:"
echo "  python3 batch_processor.py folder_with_screenshots"
echo
