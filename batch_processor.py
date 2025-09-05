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
    
    # Consolidate all contacts into one file
    all_contacts = []
    for result in all_results:
        contacts = result['results'].get('structured_contacts', [])
        for contact in contacts:
            contact['source_file'] = result['filename']
            all_contacts.append(contact)
    
    if all_contacts:
        consolidated_df = pd.DataFrame(all_contacts)
        consolidated_csv_path = output_folder / f"all_contacts_{timestamp}.csv"
        consolidated_df.to_csv(consolidated_csv_path, index=False)
        
        logger.info(f"📊 Consolidated {len(all_contacts)} contacts into {consolidated_csv_path}")
    
    logger.info(f"🎉 Batch processing complete!")
    logger.info(f"📁 Results saved to: {output_folder}")
    logger.info(f"📈 Summary: {summary_csv_path}")
    logger.info(f"📋 Detailed report: {detailed_json_path}")
    
    return detailed_report

def create_contact_database(contacts_folder, db_name="contact_database.csv"):
    """Create a searchable contact database from all extracted contacts"""
    logger = setup_logging()
    
    contacts_path = Path(contacts_folder)
    if not contacts_path.exists():
        logger.error(f"Contacts folder {contacts_folder} does not exist")
        return
    
    # Find all CSV files with contact data
    csv_files = list(contacts_path.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("batch_summary")]
    
    if not csv_files:
        logger.warning(f"No contact CSV files found in {contacts_folder}")
        return
    
    logger.info(f"Found {len(csv_files)} contact files to consolidate")
    
    all_contacts = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Add source file information
            df['source_file'] = csv_file.name
            df['extracted_date'] = datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat()
            
            all_contacts.append(df)
            logger.info(f"Added {len(df)} contacts from {csv_file.name}")
            
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {str(e)}")
    
    if not all_contacts:
        logger.error("No contacts could be loaded")
        return
    
    # Combine all contacts
    combined_df = pd.concat(all_contacts, ignore_index=True)
    
    # Clean and deduplicate
    # Remove empty rows
    combined_df = combined_df.dropna(how='all')
    
    # Remove duplicates based on email (if available)
    combined_df = combined_df.drop_duplicates(subset=['email'], keep='first')
    
    # Add unique ID
    combined_df['contact_id'] = range(1, len(combined_df) + 1)
    
    # Reorder columns
    column_order = ['contact_id', 'name', 'title', 'company', 'email', 'phone', 
                   'linkedin', 'department', 'notes', 'source', 'source_file', 'extracted_date']
    
    # Only include columns that exist
    available_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    # Save the database
    db_path = contacts_path / db_name
    combined_df.to_csv(db_path, index=False)
    
    logger.info(f"📊 Contact database created: {db_path}")
    logger.info(f"👥 Total contacts: {len(combined_df)}")
    logger.info(f"📧 Contacts with email: {combined_df['email'].notna().sum()}")
    logger.info(f"📱 Contacts with phone: {combined_df['phone'].notna().sum()}")
    
    return combined_df

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Batch process screenshots for contact extraction")
    parser.add_argument("input_folder", help="Folder containing screenshots")
    parser.add_argument("-o", "--output", help="Output folder for results")
    parser.add_argument("-d", "--database", action="store_true", 
                       help="Create consolidated contact database")
    parser.add_argument("--extensions", nargs="+", default=['.png', '.jpg', '.jpeg'],
                       help="File extensions to process")
    
    args = parser.parse_args()
    
    # Process screenshots
    if args.database:
        # If database flag is set, assume input_folder contains CSV files
        create_contact_database(args.input_folder)
    else:
        # Process image folder
        process_folder(args.input_folder, args.output, args.extensions)

if __name__ == "__main__":
    main()
