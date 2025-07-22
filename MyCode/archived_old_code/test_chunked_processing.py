#!/usr/bin/env python3
"""
Test script for chunked processing functions
This script performs basic validation of the chunked processing pipeline
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add the directory containing sctnN to Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

def test_imports():
    """Test that all required imports work correctly"""
    print("🔍 Testing imports...")
    try:
        from resonator_work import (
            get_file_duration, 
            load_chunk_data, 
            process_file_in_chunks,
            analyze_spectrograms_for_segments,
            run_chunked_snn_classification
        )
        print("✅ All resonator_work imports successful")
        
        from snn_classification import GeophoneSNN
        print("✅ SNN classification imports successful")
        
        # Test sctnN imports
        from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator
        print("✅ sctnN imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False

def test_data_files():
    """Test that required data files exist"""
    print("\n🔍 Testing data file availability...")
    
    DATA_DIR = Path.home() / "data"
    required_files = [
        DATA_DIR / "car.csv",
        DATA_DIR / "car_nothing.csv",
        DATA_DIR / "human.csv", 
        DATA_DIR / "human_nothing.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def test_analyze_spectrograms_function():
    """Test the analyze_spectrograms_for_segments function with synthetic data"""
    print("\n🔍 Testing analyze_spectrograms_for_segments function...")
    
    try:
        from resonator_work import analyze_spectrograms_for_segments
        
        # Create synthetic spectrogram data
        n_bands = 8
        n_time_bins = 1500  # 15 seconds at 100 samples/second
        
        # Create test data with some signal activity
        test_spectrogram = np.random.random((n_bands, n_time_bins)) * 0.1
        # Add some "signal" activity in specific bands
        test_spectrogram[2:5, 500:800] += 2.0  # Car-like activity
        
        # Test car signal type
        segments, labels, confidence = analyze_spectrograms_for_segments(
            test_spectrogram, 15.0, 'car'
        )
        
        print(f"✅ Car analysis: {len(segments)} segments, {np.sum(labels == 1)} with signal")
        
        # Test human signal type  
        segments, labels, confidence = analyze_spectrograms_for_segments(
            test_spectrogram, 15.0, 'human'
        )
        
        print(f"✅ Human analysis: {len(segments)} segments, {np.sum(labels == 1)} with signal")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in analyze_spectrograms_for_segments: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_duration():
    """Test getting file duration without loading full files"""
    print("\n🔍 Testing file duration estimation...")
    
    try:
        from resonator_work import get_file_duration
        
        DATA_DIR = Path.home() / "data"
        test_file = DATA_DIR / "car.csv"
        
        if test_file.exists():
            duration = get_file_duration(test_file)
            if duration is not None and duration > 0:
                print(f"✅ File duration estimation successful: {duration:.1f}s")
                return True
            else:
                print("❌ Invalid duration returned")
                return False
        else:
            print(f"⚠️  Test file not found: {test_file}")
            return True  # Not a failure if file doesn't exist
            
    except Exception as e:
        print(f"❌ Error in file duration test: {e}")
        return False

def test_snn_class():
    """Test SNN class creation and basic functionality"""
    print("\n🔍 Testing SNN class...")
    
    try:
        from snn_classification import GeophoneSNN
        
        # Create SNN instance
        snn = GeophoneSNN(n_hidden=10, learning_rate=0.01)
        print("✅ SNN instance created successfully")
        
        # Test creating network
        test_features = 56  # 8 bands × 7 features
        snn.create_network(test_features)
        print(f"✅ SNN network created with {test_features} input neurons")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in SNN class test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 CHUNKED PROCESSING VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files), 
        ("Analyze Spectrograms Test", test_analyze_spectrograms_function),
        ("File Duration Test", test_file_duration),
        ("SNN Class Test", test_snn_class)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! The chunked processing pipeline is ready to use.")
        print("\nTo run the full pipeline:")
        print("python resonator_work.py")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please fix the issues before running the full pipeline.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 