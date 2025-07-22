#!/usr/bin/env python3
"""
Chunked Processing Example for Geophone Signal Analysis
Memory-efficient processing of large files with SNN classification

This script demonstrates how to process large geophone data files
in manageable chunks to avoid memory issues while maintaining
spectrogram quality and SNN classification performance.
"""

import os
import sys
from pathlib import Path

# Add the directory containing sctnN to Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import chunked processing functions
from resonator_work import (
    run_chunked_snn_classification,
    process_file_in_chunks,
    get_file_duration,
    DATA_DIR
)

def demonstrate_chunked_processing():
    """
    Demonstrate the complete chunked processing pipeline
    """
    print("🚀 CHUNKED PROCESSING DEMONSTRATION")
    print("=" * 60)
    print("This example shows how to process large geophone files")
    print("in memory-efficient chunks for SNN classification.")
    print()
    
    # Step 1: Check file sizes
    print("📂 Step 1: Checking file sizes...")
    files_to_check = ["car.csv", "car_nothing.csv", "human.csv", "human_nothing.csv"]
    
    for filename in files_to_check:
        file_path = DATA_DIR / filename
        if file_path.exists():
            duration = get_file_duration(file_path)
            if duration:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   {filename}: {duration:.1f}s duration, {file_size_mb:.1f} MB")
            else:
                print(f"   {filename}: Could not determine duration")
        else:
            print(f"   {filename}: File not found")
    
    print()
    
    # Step 2: Demonstrate single file chunked processing
    print("🔄 Step 2: Demonstrating single file chunked processing...")
    print("Processing car.csv in 120-second chunks...")
    
    car_file = DATA_DIR / "car.csv"
    if car_file.exists():
        chunk_index = process_file_in_chunks(
            car_file, 
            chunk_duration=120,  # 2-minute chunks
            num_processes=10     # Adjust based on your system
        )
        
        if chunk_index:
            print(f"✅ Successfully processed {chunk_index['num_chunks']} chunks")
            print(f"   Total duration: {chunk_index['total_duration']:.1f}s")
            print(f"   Chunk files saved in: {os.path.dirname(chunk_index['chunk_files'][0])}")
        else:
            print("❌ Failed to process file in chunks")
    else:
        print(f"❌ File not found: {car_file}")
    
    print()
    
    # Step 3: Run complete SNN classification with chunked processing
    print("🧠 Step 3: Running complete SNN classification...")
    print("Processing both car.csv and car_nothing.csv for classification...")
    
    results = run_chunked_snn_classification(
        chunk_duration=120,  # Process in 2-minute chunks
        num_processes=10     # Adjust based on your system
    )
    
    if results:
        print(f"\n✅ CHUNKED SNN CLASSIFICATION COMPLETED!")
        print(f"   📊 Test Accuracy: {results['test_accuracy']:.1%}")
        print(f"   📁 Car chunks: {results['car_chunk_indices'][0]['num_chunks'] if results['car_chunk_indices'] else 'N/A'}")
        print(f"   🧠 SNN Architecture: {results['snn_model'].n_input_neurons} → {results['snn_model'].n_hidden} → 2 neurons")
        print(f"   💾 Model saved: chunked_snn_model.pkl")
    else:
        print("❌ SNN classification failed")
    
    print()
    print("=" * 60)
    print("📝 USAGE SUMMARY:")
    print()
    print("1. For memory-efficient processing of large files:")
    print("   from resonator_work import run_chunked_snn_classification")
    print("   results = run_chunked_snn_classification(chunk_duration=120)")
    print()
    print("2. For processing a single file in chunks:")
    print("   from resonator_work import process_file_in_chunks")
    print("   chunk_index = process_file_in_chunks('path/to/file.csv', 120)")
    print()
    print("3. Key benefits:")
    print("   • Avoids memory overflow on large files")
    print("   • Maintains spectrogram quality")
    print("   • Compatible with SNN classification")
    print("   • Parallel processing support")
    print()
    print("=" * 60)

def compare_memory_usage():
    """
    Compare memory usage between chunked and non-chunked approaches
    """
    print("\n💾 MEMORY USAGE COMPARISON")
    print("=" * 40)
    print()
    print("Traditional approach (loads entire file):")
    print("   Memory usage: O(file_size)")
    print("   Risk: Memory overflow on large files")
    print("   Processing: All at once")
    print()
    print("Chunked approach (120-second chunks):")
    print("   Memory usage: O(chunk_size) << O(file_size)")
    print("   Risk: Minimal - controlled memory usage")
    print("   Processing: Sequential chunks with cleanup")
    print()
    print("Recommended chunk sizes:")
    print("   • Small files (<10 minutes): 60-120 seconds")
    print("   • Medium files (10-60 minutes): 120-300 seconds")
    print("   • Large files (>60 minutes): 300-600 seconds")
    print()

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_chunked_processing()
    
    # Show memory usage comparison
    compare_memory_usage()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Adjust chunk_duration based on your file sizes")
    print("2. Adjust num_processes based on your system capabilities")
    print("3. Monitor chunk output directories for intermediate results")
    print("4. Use the trained SNN model for real-time classification") 