#!/usr/bin/env python3
"""
Simple runner script for the geophone processor
"""

import os
import sys

def main():
    """Run the geophone processor"""
    print("🚀 STARTING GEOPHONE PROCESSOR")
    print("=" * 50)
    
    # Check if the processor script exists
    processor_script = "process_human_file_corrected.py"
    if not os.path.exists(processor_script):
        print(f"❌ Error: {processor_script} not found")
        return False
    
    # Check if the data file exists
    data_file = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data/40Sen_30Sec_stomping_30sec_quiet.csv"
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file not found: {data_file}")
        return False
    
    print(f"✅ Processor script found: {processor_script}")
    print(f"✅ Data file found: {data_file}")
    print("=" * 50)
    
    # Run the processor
    try:
        import subprocess
        result = subprocess.run([sys.executable, processor_script], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            return True
        else:
            print(f"\n❌ Processing failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running processor: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All done! Check the chunked_output directory for results.")
    else:
        print("\n❌ Processing failed. Check the error messages above.") 