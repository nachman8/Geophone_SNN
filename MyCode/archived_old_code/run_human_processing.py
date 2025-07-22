#!/usr/bin/env python3
"""
Simple script to run human chunk processing
This is the main entry point for processing human data files only
"""

import sys
import os

# Add current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the human processing function
try:
    from process_human_chunks import process_human_files_only, test_human_processing_setup
    print("✅ Successfully imported human processing functions")
except ImportError as e:
    print(f"❌ Error importing human processing functions: {e}")
    print("Make sure process_human_chunks.py is in the same directory")
    sys.exit(1)

def main():
    """Main function to run human processing"""
    print("🚀 STARTING HUMAN CHUNK PROCESSING")
    print("=" * 50)
    print("This will process human.csv and human_nothing.csv")
    print("using the same chunked approach as your car data.")
    print()
    
    # Run setup test first
    print("🔄 TESTING SETUP...")
    print("-" * 30)
    
    try:
        test_human_processing_setup()
        
        print("\n🔄 PROCESSING HUMAN FILES...")
        print("-" * 30)
        
        # Process human files with same settings as car processing
        results = process_human_files_only(
            chunk_duration=30,  # Same as your car processing
            num_processes=15    # Same as your car processing
        )
        
        if results and results.get('status') == 'complete':
            print("\n🎉 SUCCESS! Human data processing complete!")
            print("✅ Human chunks created and saved")
            print("✅ SNN-compatible format ready")
            print("✅ Can be combined with existing car chunks")
            
            print("\n📁 OUTPUT LOCATION:")
            print("   Your human chunks are saved in:")
            print("   chunked_output_30s/human/")
            print("   chunked_output_30s/human_nothing/")
            
            print("\n�� NEXT STEPS:")
            print("   1. Combine with car chunks for full analysis")
            print("   2. Use the SNN datasets for STDP training")
            print("   3. Both car and human data now available in chunked format")
            
            return True
        else:
            print("\n❌ Human processing failed or incomplete")
            return False
            
    except Exception as e:
        print(f"⚠️  Error in human processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*50)
    print("📊 FINAL STATUS:")
    if success:
        print("✅ Human chunk processing: COMPLETE")
        print("📁 Check chunked_output_30s/ directory for results")
    else:
        print("❌ Human chunk processing: FAILED")
        print("🔧 Check error messages above for troubleshooting")
    print("=" * 50)
    
    sys.exit(0 if success else 1)
