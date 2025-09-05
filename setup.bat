@echo off
echo ===============================================
echo Job Search Contact Extractor - Windows Setup
echo ===============================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

python --version
echo.

echo [2/4] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo.

echo [3/4] Checking Tesseract OCR...
echo Please ensure Tesseract OCR is installed:
echo Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo.

echo [4/4] Testing installation...
python test_setup.py
if %errorlevel% neq 0 (
    echo SETUP FAILED: Please check the errors above
    pause
    exit /b 1
)

echo.
echo ===============================================
echo SETUP COMPLETE!
echo ===============================================
echo.
echo To start the application:
echo   streamlit run app.py
echo.
echo To test with a single image:
echo   python cli.py your_screenshot.png
echo.
echo For batch processing:
echo   python batch_processor.py folder_with_screenshots
echo.
pause
