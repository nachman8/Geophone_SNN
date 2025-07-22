#!/usr/bin/env python3
"""
Test script to verify 40-sensor CSV file loading
"""

import pandas as pd
import numpy as np
import os

def test_file_loading(file_path):
    """Test loading the 40-sensor CSV file"""
    print(f"🧪 Testing file loading: {file_path}")
    print("=" * 60)
    
    try:
        # Load the CSV file
        data = pd.read_csv(file_path, header=None)
        
        print(f"✅ File loaded successfully")
        print(f"📊 Shape: {data.shape}")
        print(f"📋 Columns: {len(data.columns)} sensors")
        print(f"📈 Rows: {len(data)} time samples")
        
        # Check duration
        sampling_freq = 1000  # Hz
        duration = len(data) / sampling_freq
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        # Show first few values from first sensor
        first_sensor = data.iloc[:, 0].values
        print(f"🔍 First sensor (column 0) sample values:")
        print(f"   First 5: {first_sensor[:5]}")
        print(f"   Last 5: {first_sensor[-5:]}")
        print(f"   Mean: {np.mean(first_sensor):.6f}")
        print(f"   Std: {np.std(first_sensor):.6f}")
        
        # Check for any NaN values
        nan_count = data.isna().sum().sum()
        print(f"🔍 NaN values: {nan_count}")
        
        # Check data types
        print(f"📋 Data types: {data.dtypes.unique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

if __name__ == "__main__":
    file_path = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data/40Sen_30Sec_stomping_30sec_quiet.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        exit(1)
    
    success = test_file_loading(file_path)
    
    if success:
        print("\n✅ File loading test passed!")
        print("The file can be processed with the fixed processor.")
    else:
        print("\n❌ File loading test failed!")
        print("Check the file format and try again.") 