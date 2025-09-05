# 👔 Job Search Contact Extractor

A powerful tool that automatically extracts HR and contact person information from job posting screenshots using OCR (Optical Character Recognition) and AI enhancement.

## 🌟 Features

- **Multi-OCR Engine Support**: Uses both Tesseract and EasyOCR for maximum text extraction accuracy
- **AI Enhancement**: Optional integration with OpenAI GPT and Google Gemini for intelligent contact parsing
- **Smart Contact Detection**: Automatically identifies emails, phone numbers, names, job titles, and company information
- **Batch Processing**: Process multiple screenshots at once
- **Data Export**: Export results to CSV and JSON formats
- **Contact Database**: Build a searchable database of all extracted contacts
- **Quality Analysis**: Get quality metrics for each extraction
- **Web Interface**: User-friendly Streamlit web interface

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows/macOS/Linux

### Required Software
- **Tesseract OCR**: Download from [GitHub](https://github.com/tesseract-ocr/tesseract)
  - Windows: Install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`

## 🚀 Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd job-helper
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables (optional)**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Configure Tesseract path (if needed)**
   - If Tesseract is not in your system PATH, update the path in your `.env` file

## 🎯 Usage

### Web Interface (Recommended)

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload screenshots** of job postings, email signatures, or contact pages

4. **View extracted contacts** in the results panel

5. **Download** CSV or JSON files with the extracted data

### Command Line (Batch Processing)

1. **Process a folder of screenshots**
   ```bash
   python batch_processor.py /path/to/screenshots -o /path/to/output
   ```

2. **Create a contact database**
   ```bash
   python batch_processor.py /path/to/csv/files --database
   ```

## 📁 File Structure

```
job-helper/
├── app.py                 # Main Streamlit application
├── batch_processor.py     # Batch processing script
├── utils.py              # Utility functions and image processing
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md           # This file
└── extracted_contacts/ # Output directory (created automatically)
    ├── *.json         # Individual extraction results
    ├── *.csv          # Structured contact data
    └── batch_*.csv    # Batch processing summaries
```

## 🔧 Configuration

### API Keys (Optional but Recommended)

For best results, add AI enhancement by setting these environment variables:

```bash
# OpenAI API Key (for GPT-based enhancement)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini API Key (for Gemini-based enhancement)
GOOGLE_API_KEY=your_google_api_key_here
```

### Tesseract Configuration

If Tesseract is not in your system PATH:

```bash
# Windows example
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# macOS example
TESSERACT_CMD=/usr/local/bin/tesseract
```

## 📊 Output Formats

### CSV Export
Contains structured contact information:
- `name`: Contact person's name
- `title`: Job title/position
- `company`: Company name
- `email`: Email address
- `phone`: Phone number
- `linkedin`: LinkedIn profile URL
- `department`: Department (HR, Recruiting, etc.)
- `source`: Extraction method used
- `notes`: Additional information

### JSON Export
Contains complete extraction results:
- Raw text from OCR engines
- Pattern matching results
- AI enhancement results
- Quality metrics

## 🎯 Best Practices

### For Better Results:
1. **Use high-quality screenshots** with clear, readable text
2. **Include contact sections** like email signatures, HR contact areas, or "Contact Us" pages
3. **Ensure good lighting** and contrast in screenshots
4. **Add API keys** for AI enhancement to get better structured results
5. **Process multiple images** of the same job posting for comprehensive contact extraction

### Supported Image Types:
- PNG, JPG, JPEG, BMP, TIFF, GIF
- Screenshots from job boards, company websites, emails
- Business cards, contact pages, email signatures

## 🔍 Features in Detail

### OCR Engines
- **Tesseract**: Industry-standard OCR with multiple configurations
- **EasyOCR**: Deep learning-based OCR for better accuracy on complex images

### Image Processing
- Automatic contrast enhancement
- Noise reduction
- Image deskewing
- Optimal resizing for OCR

### Contact Pattern Recognition
- Email address validation
- Phone number formatting (international support)
- Name extraction with validation
- Job title standardization
- Company name cleaning

### AI Enhancement
- **GPT-3.5**: Structured contact information extraction
- **Gemini**: Alternative AI-powered parsing
- Intelligent context understanding
- Better handling of complex layouts

### Quality Metrics
- Extraction completeness score
- Contact validation results
- OCR confidence levels
- Processing success rates

## 🔧 Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Install Tesseract OCR
   - Add to system PATH or configure TESSERACT_CMD

2. **Poor OCR results**
   - Try higher resolution images
   - Ensure good contrast and lighting
   - Clean up the image (remove backgrounds, enhance text)

3. **No contacts extracted**
   - Check if image contains readable text
   - Try different preprocessing options
   - Enable AI enhancement with API keys

4. **API errors**
   - Verify API keys are correct
   - Check API quotas and billing
   - Ensure internet connectivity

### Performance Tips

1. **Batch processing**: Use `batch_processor.py` for multiple files
2. **Image optimization**: Resize very large images before processing
3. **API rate limits**: Be mindful of AI service rate limits for large batches

## 📈 Advanced Usage

### Custom Processing Pipeline

```python
from app import ContactExtractor
from utils import ImagePreprocessor

# Initialize extractor
extractor = ContactExtractor()

# Custom preprocessing
preprocessor = ImagePreprocessor()
processed_images = preprocessor.preprocess_pipeline(image)

# Extract with specific settings
results = extractor.process_screenshot(processed_images['final'])
```

### Database Integration

```python
from batch_processor import create_contact_database

# Create searchable database from all extractions
database = create_contact_database('extracted_contacts/')

# Search and filter contacts
hr_contacts = database[database['title'].str.contains('HR', na=False)]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

## 🚀 Future Enhancements

- [ ] Support for more languages
- [ ] Integration with CRM systems
- [ ] Real-time processing via webcam
- [ ] Mobile app version
- [ ] Advanced duplicate detection
- [ ] Export to vCard format
- [ ] Integration with LinkedIn API
- [ ] Chrome extension for direct webpage processing

---

**Happy job hunting! 🎯**
