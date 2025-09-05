#!/usr/bin/env python3
"""
Simple CLI tool for quick contact extraction from text
Usage: python cli.py <text_file_path>
"""

import sys
import os
import argparse
from pathlib import Path
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import ContactExtractor
from utils import analyze_extraction_quality

def extract_contacts_cli(text_input, output_dir=None, show_raw=False):
    """Extract contacts from text via CLI"""
    
    # Handle file input vs direct text
    if os.path.isfile(text_input):
        text_path = Path(text_input)
        if not text_path.exists():
            print(f"❌ Error: Text file '{text_path}' not found")
            return False
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            source_name = text_path.name
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            return False
    else:
        # Direct text input
        text_content = text_input
        source_name = "direct_input"
    
    if not text_content.strip():
        print("❌ Error: No text content provided")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd() / "extracted_contacts"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Processing text: {source_name}")
    print("⏳ Initializing AI services...")
    
    try:
        # Initialize extractor
        extractor = ContactExtractor()
        
        print("� Extracting contact information...")
        results = extractor.process_text(text_content)
        
        # Analyze quality
        quality_metrics = analyze_extraction_quality(results)
        
        # Save results
        filename = source_name.replace('.', '_') if '.' in source_name else source_name
        json_path, csv_path = extractor.save_results(results, filename)
        
        # Move to output directory
        if json_path:
            new_json_path = output_dir / json_path.name
            json_path.rename(new_json_path)
            json_path = new_json_path
        
        if csv_path:
            new_csv_path = output_dir / csv_path.name
            csv_path.rename(new_csv_path)
            csv_path = new_csv_path
        
        # Display results
        print("\n" + "="*60)
        print("📊 EXTRACTION RESULTS")
        print("="*60)
        
        contacts = results.get('structured_contacts', [])
        
        if contacts:
            print(f"✅ Found {len(contacts)} contact(s)")
            print(f"📈 Quality Score: {quality_metrics['quality_score']:.1f}/100")
            print()
            
            for i, contact in enumerate(contacts, 1):
                print(f"👤 Contact {i}:")
                if contact.get('name'):
                    print(f"   Name: {contact['name']}")
                if contact.get('title'):
                    print(f"   Title: {contact['title']}")
                if contact.get('company'):
                    print(f"   Company: {contact['company']}")
                if contact.get('email'):
                    print(f"   📧 Email: {contact['email']}")
                if contact.get('phone'):
                    print(f"   📱 Phone: {contact['phone']}")
                if contact.get('linkedin'):
                    print(f"   🔗 LinkedIn: {contact['linkedin']}")
                if contact.get('department'):
                    print(f"   Department: {contact['department']}")
                if contact.get('source'):
                    print(f"   Source: {contact['source']}")
                print()
        else:
            print("❌ No contacts found")
            
            # Show pattern analysis if available
            patterns = results.get('contact_patterns', {})
            found_something = False
            
            for category, items in patterns.items():
                if items:
                    found_something = True
                    print(f"📋 Found {category.replace('_', ' ')}: {', '.join(items)}")
            
            if not found_something:
                print("🔍 No contact patterns detected")
        
        # Show raw text if requested
        if show_raw:
            print("\n" + "="*60)
            print("📝 RAW TEXT INPUT")
            print("="*60)
            print(text_content[:1000] + "..." if len(text_content) > 1000 else text_content)
        
        # Show file locations
        print("\n" + "="*60)
        print("💾 SAVED FILES")
        print("="*60)
        
        if json_path:
            print(f"📄 JSON: {json_path}")
        if csv_path:
            print(f"📊 CSV: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing text: {str(e)}")
        return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Extract contact information from text files or job posting text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py job_posting.txt
  python cli.py "Sarah Johnson, HR Manager at TechCorp..." --output ./results --raw
  python cli.py contact_info.md --raw
        """
    )
    
    parser.add_argument(
        'text_input',
        help='Path to text file or direct text input'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for results (default: ./extracted_contacts)',
        default=None
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Show raw input text'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Job Contact Extractor CLI v2.0 (Text-based)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the text
    success = extract_contacts_cli(
        text_input=args.text_input,
        output_dir=args.output,
        show_raw=args.raw
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
