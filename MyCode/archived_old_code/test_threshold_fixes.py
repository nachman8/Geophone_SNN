#!/usr/bin/env python3
"""
Test Script: Validate Fixed Threshold Detection
"""

import sys
import os
from pathlib import Path

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import the fixed functions
from load_saved_chunks import train_from_saved_chunks, load_chunks_directly, prepare_binary_training_data

def test_threshold_fixes():
    """
    Test the fixed threshold detection
    """
    print("🧪 TESTING FIXED THRESHOLD DETECTION")
    print("=" * 50)
    
    # Set the chunks directory
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    print("🔄 Loading saved chunks with fixed thresholds...")
    chunk_data = load_chunks_directly(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return False
    
    print(f"✅ Loaded chunks for {list(chunk_data.keys())}")
    
    # Test Human Classification with fixed thresholds
    print(f"\n👤 TESTING HUMAN vs HUMAN_NOTHING WITH FIXED THRESHOLDS")
    
    if 'human' in chunk_data and 'human_nothing' in chunk_data:
        human_segments, human_labels = prepare_binary_training_data(chunk_data, 'human')
        
        if len(human_segments) > 0:
            signal_count = sum(1 for label in human_labels if label == 0)  # 0 = signal
            nothing_count = sum(1 for label in human_labels if label == 1)  # 1 = nothing
            
            print(f"📊 HUMAN CLASSIFICATION RESULTS (WITH FIXES):")
            print(f"   Total segments: {len(human_segments)}")
            print(f"   Signal segments (class 0): {signal_count}")
            print(f"   Nothing segments (class 1): {nothing_count}")
            
            if nothing_count > 0:
                print(f"   ✅ SUCCESS: Found both signal and nothing segments!")
                print(f"   📈 Class balance: {signal_count}:{nothing_count}")
                success = True
            else:
                print(f"   ❌ STILL FAILING: No nothing segments found")
                success = False
        else:
            print(f"   ❌ No human segments found")
            success = False
    else:
        print(f"   ❌ Missing human chunk data")
        success = False
    
    # Test Car Classification  
    print(f"\n🚗 TESTING CAR vs CAR_NOTHING")
    
    if 'car' in chunk_data and 'car_nothing' in chunk_data:
        car_segments, car_labels = prepare_binary_training_data(chunk_data, 'car')
        
        if len(car_segments) > 0:
            signal_count = sum(1 for label in car_labels if label == 0)  # 0 = signal
            nothing_count = sum(1 for label in car_labels if label == 1)  # 1 = nothing
            
            print(f"📊 CAR CLASSIFICATION RESULTS:")
            print(f"   Total segments: {len(car_segments)}")
            print(f"   Signal segments (class 0): {signal_count}")
            print(f"   Nothing segments (class 1): {nothing_count}")
            
            if nothing_count > 0 and signal_count > 0:
                print(f"   ✅ SUCCESS: Good class balance!")
                print(f"   📈 Class balance: {signal_count}:{nothing_count}")
            else:
                print(f"   ⚠️  WARNING: Imbalanced classes")
        else:
            print(f"   ❌ No car segments found")
    else:
        print(f"   ❌ Missing car chunk data")
    
    return success

def run_quick_snn_test():
    """
    Run a quick SNN test if thresholds are working
    """
    print(f"\n🧠 QUICK SNN CLASSIFICATION TEST")
    print("=" * 35)
    
    try:
        print("🔄 Attempting to train SNN with fixed thresholds...")
        
        # Set the chunks directory  
        chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
        
        # Run training with fixed thresholds
        results = train_from_saved_chunks(chunks_dir)
        
        if results:
            print(f"\n🎉 SNN TRAINING RESULTS WITH FIXED THRESHOLDS:")
            
            if results.get('car'):
                car_res = results['car']
                print(f"\n🚗 CAR CLASSIFICATION:")
                print(f"   ✅ Accuracy: {car_res['accuracy']:.1%}")
                print(f"   💾 Model: {car_res['model_path']}")
            
            if results.get('human'):
                human_res = results['human']
                print(f"\n👤 HUMAN CLASSIFICATION:")
                print(f"   ✅ Accuracy: {human_res['accuracy']:.1%}")
                print(f"   💾 Model: {human_res['model_path']}")
                print(f"   🎯 THRESHOLD FIX: SUCCESS!")
            else:
                print(f"\n👤 HUMAN CLASSIFICATION:")
                print(f"   ❌ Still failing - need further threshold adjustment")
        else:
            print("❌ SNN training failed")
            
    except Exception as e:
        print(f"❌ Error during SNN test: {e}")
        print("   This may indicate threshold fixes need further refinement")

if __name__ == "__main__":
    print("🔧 THRESHOLD FIX VALIDATION TEST")
    print("Testing whether the fixed adaptive thresholds solve the classification issues")
    print("=" * 70)
    
    # Test threshold fixes
    threshold_success = test_threshold_fixes()
    
    if threshold_success:
        print(f"\n✅ THRESHOLD FIXES SUCCESSFUL!")
        print("   The human_nothing file now produces mixed signal/nothing segments")
        print("   SNN classification should now work for both car and human data")
        
        # Run quick SNN test
        run_quick_snn_test()
    else:
        print(f"\n❌ THRESHOLD FIXES NEED FURTHER ADJUSTMENT")
        print("   The human_nothing file still classifies all segments as signal")
        print("   Consider making the 'nothing' thresholds even more conservative")
    
    print("\n" + "=" * 70)
    print("💡 NEXT STEPS:")
    if threshold_success:
        print("• Thresholds are working - proceed with full SNN training")
        print("• Monitor classification accuracy and fine-tune if needed")
        print("• Consider cross-validation for robust performance measurement")
    else:
        print("• Increase nothing thresholds further (activity_ratio > 0.45, signal_strength > 4.0x)")
        print("• Analyze human_nothing file characteristics more deeply")
        print("• Consider different detection strategies for human vs car signals")
    print("=" * 70) 