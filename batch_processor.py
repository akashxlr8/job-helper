import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import argparse
from PIL import Image
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import ContactExtractor
from utils import analyze_extraction_quality
from database import save_contacts_to_db

def setup_logging():
    """Set up logging for batch processing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_folder(input_folder, output_folder=None, file_extensions=None):
    """Process all images in a folder"""
    logger = setup_logging()
    
    if file_extensions is None:
        file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    
    input_path = Path(input_folder)
    if not input_path.exists():
        logger.error(f"Input folder {input_folder} does not exist")
        return
    
    if output_folder is None:
        output_folder = input_path / "extracted_contacts"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(exist_ok=True)
    
    # Initialize extractor
    logger.info("Initializing contact extractor...")
    extractor = ContactExtractor()
    
    # Find all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No image files found in {input_folder}")
        return
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Process each image
    all_results = []
    summary_data = []
    
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load and process image
            image = Image.open(image_file)
            results = extractor.process_screenshot(image)
            
            # Save individual results
            filename = image_file.stem
            json_path, csv_path = extractor.save_results(results, filename)

            # Save to database
            try:
                save_contacts_to_db(results, datetime.now().strftime('%Y%m%d_%H%M%S'), image_file.name)
                logger.info(f"✅ Results for {image_file.name} saved to the database.")
            except Exception as e:
                logger.error(f"❌ Error saving results for {image_file.name} to database: {str(e)}")
            
            # Move files to output folder
            if json_path:
                new_json_path = output_folder / json_path.name
                json_path.rename(new_json_path)
                
            if csv_path:
                new_csv_path = output_folder / csv_path.name
                csv_path.rename(new_csv_path)
            
            # Analyze quality
            quality_metrics = analyze_extraction_quality(results)
            
            # Add to summary
            summary_data.append({
                'filename': image_file.name,
                'total_contacts': quality_metrics['total_contacts'],
                'contacts_with_email': quality_metrics['contacts_with_email'],
                'contacts_with_phone': quality_metrics['contacts_with_phone'],
                'quality_score': quality_metrics['quality_score'],
                'processed_at': datetime.now().isoformat()
            })
            
            all_results.append({
                'filename': image_file.name,
                'results': results,
                'quality_metrics': quality_metrics
            })
            
            logger.info(f"✅ Processed {image_file.name} - Found {quality_metrics['total_contacts']} contacts")
            
        except Exception as e:
            logger.error(f"❌ Error processing {image_file.name}: {str(e)}")
            summary_data.append({
                'filename': image_file.name,
                'total_contacts': 0,
                'contacts_with_email': 0,
                'contacts_with_phone': 0,
                'quality_score': 0,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            })
    
    # Create summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_folder / f"batch_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Save detailed JSON report
    detailed_report = {
        'batch_info': {
            'input_folder': str(input_folder),
            'output_folder': str(output_folder),
            'processed_at': datetime.now().isoformat(),
            'total_files': len(image_files),
            'successful_extractions': len([r for r in all_results if r['quality_metrics']['total_contacts'] > 0])
        },
        'summary_statistics': {
            'total_contacts_found': sum(r['quality_metrics']['total_contacts'] for r in all_results),
            'average_quality_score': sum(r['quality_metrics']['quality_score'] for r in all_results) / len(all_results) if all_results else 0,
            'files_with_contacts': len([r for r in all_results if r['quality_metrics']['total_contacts'] > 0])
        },
        'detailed_results': all_results
    }
    
    detailed_json_path = output_folder / f"detailed_report_{timestamp}.json"
    with open(detailed_json_path, 'w') as f:
        json.dump(detailed_report, f, indent=2, default=str)
    
    # Save to database
            try:
                save_contacts_to_db(results, datetime.now().strftime('%Y%m%d_%H%M%S'), image_file.name)
                logger.info(f"✅ Results for {image_file.name} saved to the database.")
            except Exception as e:
                logger.error(f"❌ Error saving results for {image_file.name} to database: {str(e)}")
    
    logger.info(f"🎉 Batch processing complete!")
    logger.info(f"📁 Results saved to: {output_folder}")
    logger.info(f"📈 Summary: {summary_csv_path}")
    logger.info(f"📋 Detailed report: {detailed_json_path}")
    
    return detailed_report

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Batch process screenshots for contact extraction")
    parser.add_argument("input_folder", help="Folder containing screenshots")
    parser.add_argument("-o", "--output", help="Output folder for results")
    parser.add_argument("--extensions", nargs="+", default=['.png', '.jpg', '.jpeg'],
                       help="File extensions to process")
    
    args = parser.parse_args()
    
    # Process image folder
    process_folder(args.input_folder, args.output, args.extensions)

if __name__ == "__main__":
    main()
