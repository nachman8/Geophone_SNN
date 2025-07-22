#!/usr/bin/env python3
"""
Chunk Data Structure Inspector
=============================

This script examines the structure of the chunk pickle files to understand
how the resonator output data is organized.
"""

import pickle
import numpy as np
from pathlib import Path

CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

def inspect_chunk_data():
    """Inspect the structure of chunk data files"""
    print("🔍 CHUNK DATA STRUCTURE INSPECTOR")
    print("=" * 50)
    
    # Check each category
    categories = ['human', 'human_nothing', 'car', 'car_nothing']
    
    for category in categories:
        print(f"\n📂 Category: {category}")
        category_dir = Path(CHUNKED_OUTPUT_DIR) / category
        
        # Get first chunk as sample
        chunk_dirs = [d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith('chunk_')]
        
        if chunk_dirs:
            sample_chunk = chunk_dirs[0]
            pkl_file = sample_chunk / f"{sample_chunk.name}_data.pkl"
            
            if pkl_file.exists():
                print(f"  📄 Examining: {pkl_file}")
                
                try:
                    with open(pkl_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    print(f"  📊 Data type: {type(chunk_data)}")
                    
                    if isinstance(chunk_data, dict):
                        print(f"  🔑 Keys: {list(chunk_data.keys())}")
                        
                        for key, value in chunk_data.items():
                            print(f"    {key}:")
                            print(f"      Type: {type(value)}")
                            
                            if hasattr(value, '__len__'):
                                print(f"      Length: {len(value)}")
                                
                                if isinstance(value, list) and len(value) > 0:
                                    print(f"      First element type: {type(value[0])}")
                                    if isinstance(value[0], np.ndarray):
                                        print(f"      First element shape: {value[0].shape}")
                                        print(f"      First element sample: {value[0][:10] if len(value[0]) > 0 else 'empty'}")
                                        
                                        # Examine all elements
                                        lengths = [len(arr) for arr in value if isinstance(arr, np.ndarray)]
                                        print(f"      All spike lengths: {lengths}")
                                        
                    elif isinstance(chunk_data, list):
                        print(f"  📋 List length: {len(chunk_data)}")
                        if len(chunk_data) > 0:
                            print(f"  📋 First element type: {type(chunk_data[0])}")
                    
                    elif isinstance(chunk_data, np.ndarray):
                        print(f"  🔢 Array shape: {chunk_data.shape}")
                        print(f"  🔢 Array dtype: {chunk_data.dtype}")
                        print(f"  🔢 Sample values: {chunk_data.flat[:10]}")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {pkl_file}: {e}")
                    
                # Also check index file if it exists
                index_file = category_dir / "chunk_index.pkl"
                if index_file.exists():
                    try:
                        with open(index_file, 'rb') as f:
                            index_data = pickle.load(f)
                        print(f"  📇 Index file type: {type(index_data)}")
                        if isinstance(index_data, dict):
                            print(f"  📇 Index keys: {list(index_data.keys())}")
                    except Exception as e:
                        print(f"  ❌ Error loading index: {e}")
            else:
                print(f"  ❌ No data file found in {sample_chunk}")
        else:
            print(f"  ❌ No chunk directories found in {category}")

if __name__ == "__main__":
    inspect_chunk_data() 