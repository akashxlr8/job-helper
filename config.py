# Job Search Contact Extractor Configuration

## OCR Settings
OCR_CONFIDENCE_THRESHOLD = 0.3
OCR_PREPROCESSING_MODES = ['original', 'enhanced', 'threshold', 'morph']

## Contact Pattern Settings
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 50
MIN_PHONE_DIGITS = 10

## Export Settings
DEFAULT_OUTPUT_FORMAT = 'csv'
INCLUDE_CONFIDENCE_SCORES = True
SAVE_RAW_TEXT = True

## AI Enhancement Settings
OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_TEMPERATURE = 0.1
GEMINI_MODEL = 'gemini-pro'
MAX_TOKENS_PER_REQUEST = 4000

## Quality Thresholds
MIN_QUALITY_SCORE = 20
WARN_QUALITY_SCORE = 50

## File Processing
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
MAX_FILE_SIZE_MB = 10
AUTO_RESIZE_LARGE_IMAGES = True
TARGET_IMAGE_HEIGHT = 800
