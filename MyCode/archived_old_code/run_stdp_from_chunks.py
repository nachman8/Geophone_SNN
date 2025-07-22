#!/usr/bin/env python3
"""
Simple runner script for STDP classification from existing chunks
This loads your saved chunks and creates STDP datasets without re-processing
"""

import sys
import os

# Add current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """Main function to run STDP classification from chunks"""
    print("🧠 STDP CLASSIFICATION FROM EXISTING CHUNKS")
    print("=" * 60)
    print("Loading your existing processed chunks and creating")
    print("STDP classification datasets (no re-processing needed)")
    print()
    
    try:
        # Import the main function
        from load_chunks_and_classify import run_stdp_classification_from_chunks
        
        print("🔄 LOADING CHUNKS AND CREATING STDP DATASETS...")
        print("-" * 50)
        
        # Run the STDP classification pipeline
        results = run_stdp_classification_from_chunks()
        
        if results and results.get('status') == 'ready_for_stdp_training':
            print("\n🎉 SUCCESS! STDP CLASSIFICATION READY!")
            print("=" * 50)
            
            summary = results['summary']
            print(f"🚗 Car data: {summary['car_data']['total_duration_s']:.1f}s")
            print(f"👤 Human data: {summary['human_data']['total_duration_s']:.1f}s")
            print(f"🎯 Training frequencies: {len(results['available_frequencies'])}")
            
            print(f"\n💾 DATASETS SAVED:")
            if 'car_dataset_path' in summary['training_ready']:
                print(f"   🚗 Car: {summary['training_ready']['car_dataset_path']}")
            if 'human_dataset_path' in summary['training_ready']:
                print(f"   👤 Human: {summary['training_ready']['human_dataset_path']}")
            
            print(f"\n🚀 READY FOR STDP TRAINING!")
            print("   Your data is now compatible with old notebook approaches")
            print("   Load the spike trains and start STDP learning")
            
            return True
            
        elif results:
            print(f"\n⚠️  Partial success: {results.get('status', 'unknown')}")
            return False
            
        else:
            print(f"\n❌ Failed to create STDP datasets")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure load_chunks_and_classify.py is in the same directory")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*60)
    print("📊 FINAL STATUS:")
    if success:
        print("✅ STDP datasets created successfully")
        print("🧠 Ready for STDP training!")
    else:
        print("❌ STDP dataset creation failed")
        print("🔧 Check error messages above")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
