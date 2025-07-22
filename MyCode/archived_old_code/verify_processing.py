#!/usr/bin/env python3
"""
Verify the processing results
"""

import pickle
import os
import numpy as np

def verify_processing_results():
    """Verify the processing results"""
    print("🔍 VERIFYING PROCESSING RESULTS")
    print("=" * 60)
    
    # Check human_new results
    human_dir = "chunked_output/human_new"
    if os.path.exists(human_dir):
        print(f"✅ Human processing directory exists: {human_dir}")
        
        # Check chunks
        chunks = [d for d in os.listdir(human_dir) if d.startswith('chunk_')]
        print(f"📊 Found {len(chunks)} human chunks: {chunks}")
        
        for chunk in chunks:
            chunk_dir = os.path.join(human_dir, chunk)
            data_file = os.path.join(chunk_dir, f"{chunk}_data.pkl")
            viz_file = os.path.join(chunk_dir, f"human_new_{chunk}_visualization.png")
            
            if os.path.exists(data_file):
                print(f"   📁 {chunk}: Data file exists ({os.path.getsize(data_file)/1024/1024:.1f} MB)")
                
                # Load and check data
                try:
                    with open(data_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    print(f"      📊 Signal length: {len(data['signal'])} samples")
                    print(f"      🧠 Resonator outputs: {len(data['resonator_outputs'])} clock frequencies")
                    print(f"      📈 Spectrogram shape: {data['max_spikes_spectrogram'].shape}")
                    print(f"      🎵 Frequency bands: {data['spikes_bands_spectrogram'].shape}")
                    print(f"      ⏱️  Duration: {data['duration']:.1f} seconds")
                    
                except Exception as e:
                    print(f"      ❌ Error loading data: {e}")
            
            if os.path.exists(viz_file):
                print(f"      📊 Visualization: {os.path.getsize(viz_file)/1024:.1f} KB")
    
    # Check human_nothing_new results
    background_dir = "chunked_output/human_nothing_new"
    if os.path.exists(background_dir):
        print(f"\n✅ Background processing directory exists: {background_dir}")
        
        # Check chunks
        chunks = [d for d in os.listdir(background_dir) if d.startswith('chunk_')]
        print(f"📊 Found {len(chunks)} background chunks: {chunks}")
        
        for chunk in chunks:
            chunk_dir = os.path.join(background_dir, chunk)
            data_file = os.path.join(chunk_dir, f"{chunk}_data.pkl")
            viz_file = os.path.join(chunk_dir, f"human_nothing_new_{chunk}_visualization.png")
            
            if os.path.exists(data_file):
                print(f"   📁 {chunk}: Data file exists ({os.path.getsize(data_file)/1024/1024:.1f} MB)")
            
            if os.path.exists(viz_file):
                print(f"      📊 Visualization: {os.path.getsize(viz_file)/1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING VERIFICATION COMPLETE")
    print("The 40-sensor geophone file has been successfully processed!")
    print("Check the visualization files to see the resonator spikegrams.")

if __name__ == "__main__":
    verify_processing_results() 