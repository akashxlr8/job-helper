# 🚀 Quick Start Guide - Job Search Contact Extractor

## What This Tool Does
Automatically extracts HR and contact person information from screenshots of:
- Job postings (LinkedIn, Indeed, company websites)
- Email signatures 
- Company contact pages
- Business cards
- Any image containing contact information

## ⚡ Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract  
# Ubuntu: sudo apt install tesseract-ocr
```

### Step 2: Test Installation
```bash
python test_setup.py
```

### Step 3: Start the App
```bash
streamlit run app.py
```

## 🎯 Usage Examples

### Example 1: Single Screenshot
```bash
python cli.py job_posting.png
```

### Example 2: Batch Processing
```bash
python batch_processor.py ./screenshots_folder
```

### Example 3: Web Interface
1. Run `streamlit run app.py`
2. Upload screenshots in the web interface
3. Download CSV files with contact data

## 📊 What You'll Get

For each screenshot, the tool extracts:
- ✅ Names (HR managers, recruiters, contacts)
- ✅ Email addresses (validated)
- ✅ Phone numbers (formatted)
- ✅ Job titles and departments
- ✅ Company information
- ✅ LinkedIn profiles
- ✅ Quality scores and confidence metrics

## 🔧 Optional: AI Enhancement

For better results, add API keys to `.env` file:
```bash
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## 📁 Output Files

- **CSV**: Structured contact data for spreadsheets
- **JSON**: Complete extraction results with metadata
- **Batch Summary**: Overview of processing results

## 💡 Tips for Best Results

1. **High-quality screenshots**: Clear, readable text
2. **Include contact sections**: HR areas, email signatures
3. **Multiple angles**: Screenshot different parts of the same page
4. **Add AI keys**: For intelligent parsing and structure

## 🆘 Common Issues

**"Tesseract not found"**
- Install Tesseract OCR from the links above
- Add to system PATH

**"No contacts found"**
- Check image quality and text clarity
- Try the web interface for better preprocessing
- Enable AI enhancement with API keys

**"Import errors"**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## 📚 Need More Help?

- Run `python demo.py` to see what the tool can do
- Check `README.md` for detailed documentation
- Use `python test_setup.py` to diagnose issues

---

**🎉 Happy job hunting! This tool will save you hours of manual contact research.**
