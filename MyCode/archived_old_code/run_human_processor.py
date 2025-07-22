#!/usr/bin/env python3
"""
Simple Usage Script for Human Stepping File Processing
=====================================================

This script provides an easy way to process your 60-second geophone file.
Just update the file_path below and run this script.
"""

import os
import sys
from process_human_file_simple import process_human_stepping_file

def main():
    # =====================================================
    # UPDATE THIS PATH TO YOUR 60-SECOND GEOPHONE FILE
    # =====================================================
    file_path = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data/40Sen_30Sec_stomping_30sec_quiet.csv"
    
    # You can also get the file path from command line argument
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(file_path) or "/path/to/" in file_path:
        print("❌ Please update the file_path variable or provide file path as argument")
        print("\nOptions:")
        print("1. Edit this file and update the file_path variable")
        print("2. Run: python run_human_processor.py /your/actual/file/path.csv")
        print("\nExample file paths:")
        print("  - /home/user/data/geophone_data.csv")
        print("  - ./geophone_60sec.csv")
        print("  - /data/seismic/human_stepping_60s.csv")
        return
    
    print("🧠 STARTING HUMAN STEPPING FILE PROCESSING")
    print("=" * 60)
    print(f"📂 Input file: {file_path}")
    
    # Verify file format
    try:
        import pandas as pd
        data = pd.read_csv(file_path, nrows=5)
        print(f"📊 File format check: ✅ Valid CSV")
        print(f"📋 Columns found: {list(data.columns)}")
        
        if 'amplitude' not in data.columns and len(data.columns) < 2:
            print("⚠️  Warning: Expected 'amplitude' column or at least 2 columns")
            print("   The script will use the second column as signal data")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print("=" * 60)
    
    # Process the file
    try:
        summary = process_human_stepping_file(
            file_path=file_path,
            human_end_time=32.5,  # Human stepping ends at 32.5 seconds
            chunk_duration=30,    # Process in 30-second chunks
            num_processes=15      # Use 15 parallel processes (adjust if needed)
        )
        
        if summary:
            print("\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("📁 Output locations:")
            print(f"   👤 Human stepping chunks: {summary['output_directories']['human']}")
            print(f"   🔇 Background chunks: {summary['output_directories']['background']}")
            print(f"   📋 Processing summary: {summary.get('summary_file', 'processing_summary.pkl')}")
            
            print(f"\n📊 Results summary:")
            print(f"   ⏱️  Processing time: {summary['processing_time']:.1f} seconds")
            print(f"   👤 Human chunks: {summary['human_chunks']}")
            print(f"   🔇 Background chunks: {summary['background_chunks']}")
            
            print(f"\n📂 Directory structure created:")
            print(f"   chunked_output/")
            print(f"   ├── human_new/")
            print(f"   │   ├── chunk_0/")
            print(f"   │   │   ├── chunk_0_data.pkl")
            print(f"   │   │   └── human_new_chunk_0_visualization.png")
            print(f"   │   └── chunk_1/ (if applicable)")
            print(f"   ├── human_nothing_new/")
            print(f"   │   ├── chunk_0/")
            print(f"   │   │   ├── chunk_0_data.pkl")
            print(f"   │   │   └── human_nothing_new_chunk_0_visualization.png")
            print(f"   │   └── chunk_1/ (if applicable)")
            print(f"   └── processing_summary.pkl")
            
            print(f"\n🎯 Next steps:")
            print(f"   1. Check the visualization images to verify processing")
            print(f"   2. The .pkl files contain the spikegram data for analysis")
            print(f"   3. These directories can now be used with the ensemble SNN training")
            
        else:
            print("\n❌ Processing failed. Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("🔧 Troubleshooting tips:")
        print("   1. Check if the file path is correct")
        print("   2. Ensure the file is a valid CSV with geophone data")
        print("   3. Make sure you have enough disk space")
        print("   4. Try reducing num_processes if you get memory errors")

if __name__ == "__main__":
    main() 