#!/usr/bin/env python3
"""
Quick test script to verify STDP classification from chunks will work
"""

import os
import sys
from pathlib import Path

def test_chunk_setup():
    """Test that chunks are set up correctly for STDP classification"""
    print("🧪 TESTING STDP CHUNK SETUP")
    print("=" * 40)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check for chunks directory
    chunks_dir = Path("project/MyCode/chunked_output_30s")
    if not chunks_dir.exists():
        chunks_dir = Path("chunked_output_30s")
    
    if not chunks_dir.exists():
        print("❌ chunked_output_30s directory not found!")
        print("   Expected locations:")
        print("   - project/MyCode/chunked_output_30s")
        print("   - chunked_output_30s")
        return False
    
    print(f"✅ Found chunks directory: {chunks_dir}")
    
    # Check chunk subdirectories
    expected_dirs = ['car', 'car_nothing', 'human', 'human_nothing']
    found_dirs = []
    
    for dirname in expected_dirs:
        chunk_subdir = chunks_dir / dirname
        if chunk_subdir.exists():
            found_dirs.append(dirname)
            # Check for chunk index
            index_file = chunk_subdir / "chunk_index.pkl"
            chunk_count = len([d for d in chunk_subdir.iterdir() if d.is_dir() and d.name.startswith('chunk_')])
            
            if index_file.exists():
                print(f"✅ {dirname}: {chunk_count} chunks, index file present")
            else:
                print(f"⚠️  {dirname}: {chunk_count} chunks, no index file")
        else:
            print(f"❌ {dirname}: directory not found")
    
    print(f"\n📊 Summary: Found {len(found_dirs)}/{len(expected_dirs)} chunk directories")
    
    # Check for required files
    print(f"\n🔍 CHECKING REQUIRED FILES:")
    required_files = [
        "project/MyCode/resonator_work.py",
        "project/MyCode/load_chunks_and_classify.py",
        "project/MyCode/run_stdp_from_chunks.py"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {Path(file_path).name}")
        else:
            print(f"❌ {Path(file_path).name}")
            all_files_exist = False
    
    # Test import
    print(f"\n🧪 TESTING IMPORTS:")
    try:
        sys.path.insert(0, str(Path("project/MyCode").absolute()))
        print("   Testing resonator_work import...")
        import resonator_work
        print("✅ resonator_work imported successfully")
        
        print("   Testing load_chunks_and_classify import...")
        import load_chunks_and_classify
        print("✅ load_chunks_and_classify imported successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        all_files_exist = False
    
    # Final assessment
    print(f"\n🎯 READINESS ASSESSMENT:")
    if len(found_dirs) >= 2 and all_files_exist:
        print("✅ READY FOR STDP CLASSIFICATION!")
        print("   You can run: python project/MyCode/run_stdp_from_chunks.py")
        return True
    else:
        print("❌ NOT READY - missing requirements")
        print("   Check the issues listed above")
        return False

if __name__ == "__main__":
    print("🧪 STDP CHUNKS READINESS TEST")
    print("=" * 50)
    print("This test verifies that your chunks are ready")
    print("for STDP classification without re-processing.")
    print()
    
    success = test_chunk_setup()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST PASSED - Ready for STDP classification!")
        print("\n💡 NEXT STEPS:")
        print("1. Run: python project/MyCode/run_stdp_from_chunks.py")
        print("2. Wait for STDP datasets to be created")
        print("3. Use the datasets for STDP training")
    else:
        print("❌ TEST FAILED - Requirements not met")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure you're in the correct directory")
        print("2. Verify chunks were processed successfully")
        print("3. Check that all required files exist")
    
    print("=" * 50)
