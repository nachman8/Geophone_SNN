#!/usr/bin/env python3
"""
Debug Feature Dimensions
========================

This script investigates why the feature dimensions are so large
and how to properly extract discriminative features.
"""

import pickle
import numpy as np
from pathlib import Path

CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

def debug_features():
    """Debug the feature structure"""
    print("🔬 DEBUGGING FEATURE DIMENSIONS")
    print("=" * 50)
    
    # Check one file from each category
    categories = ['human', 'car']
    
    for category in categories:
        print(f"\n📂 Category: {category}")
        category_dir = Path(CHUNKED_OUTPUT_DIR) / category
        
        # Get first chunk
        chunk_dirs = [d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith('chunk_')]
        if chunk_dirs:
            chunk_dir = chunk_dirs[0]
            pkl_file = chunk_dir / f"{chunk_dir.name}_data.pkl"
            
            with open(pkl_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            print(f"  📄 File: {pkl_file}")
            
            # Examine each feature type in detail
            for key in ['max_spikes_spectrogram', 'spikes_bands_spectrogram', 'signal']:
                if key in chunk_data:
                    value = chunk_data[key]
                    print(f"    {key}:")
                    print(f"      Type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"      Shape: {value.shape}")
                        print(f"      Dtype: {value.dtype}")
                        print(f"      Total elements: {value.size}")
                        print(f"      Sample values: {value.flat[:10]}")
                        if value.ndim > 1:
                            print(f"      Flattened size: {value.flatten().size}")
                    elif hasattr(value, '__len__'):
                        print(f"      Length: {len(value)}")
                        if len(value) > 0:
                            print(f"      First element: {type(value[0])}, value: {value[0]}")
            
            # Calculate expected feature size breakdown
            print(f"  📊 Feature size breakdown:")
            total_features = 0
            
            if 'max_spikes_spectrogram' in chunk_data:
                mss = chunk_data['max_spikes_spectrogram']
                if isinstance(mss, np.ndarray):
                    mss_size = mss.size
                    total_features += mss_size
                    print(f"    max_spikes_spectrogram: {mss_size}")
            
            if 'spikes_bands_spectrogram' in chunk_data:
                sbs = chunk_data['spikes_bands_spectrogram']
                if isinstance(sbs, np.ndarray):
                    sbs_size = sbs.size
                    total_features += sbs_size
                    print(f"    spikes_bands_spectrogram: {sbs_size}")
            
            # Signal stats
            signal_stats = 5
            total_features += signal_stats
            print(f"    signal_statistics: {signal_stats}")
            print(f"    TOTAL: {total_features}")

if __name__ == "__main__":
    debug_features() 