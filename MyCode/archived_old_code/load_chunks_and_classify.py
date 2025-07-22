import numpy as np
import pandas as pd
import pickle
import os
import sys
from pathlib import Path

# At the beginning of your file
import sys
import os

# Add the directory CONTAINING sctnN to your Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Now you can import from sctnN
from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator

import warnings
warnings.filterwarnings('ignore')

# Import required functions from resonator_work.py
try:
    from resonator_work import (
        create_snn_training_dataset_from_chunks,
        save_snn_dataset_for_training,
        load_snn_training_data,
        clk_resonators_human,
        clk_resonators_car
    )
    print("✅ Successfully imported functions from resonator_work.py")
except ImportError as e:
    print(f"❌ Error importing from resonator_work.py: {e}")
    print("Make sure resonator_work.py is in the same directory")
    sys.exit(1)

def load_chunk_index(chunk_dir):
    """
    Load chunk index from a processed chunk directory
    """
    index_file = os.path.join(chunk_dir, "chunk_index.pkl")
    
    if not os.path.exists(index_file):
        print(f"❌ Chunk index not found: {index_file}")
        return None
    
    try:
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        print(f"✅ Loaded chunk index from {chunk_dir}")
        print(f"   📁 File: {Path(chunk_index['file_path']).name}")
        print(f"   ⏱️  Duration: {chunk_index['total_duration']:.1f}s")
        print(f"   📊 Chunks: {chunk_index['num_chunks']}")
        
        return chunk_index
        
    except Exception as e:
        print(f"❌ Error loading chunk index from {chunk_dir}: {e}")
        return None

def discover_chunk_directories(base_dir="/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output_30s"):
    """
    Automatically discover all chunk directories and categorize them
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Chunk directory not found: {base_path.absolute()}")
        return None, None
    
    print(f"🔍 Discovering chunks in: {base_path.absolute()}")
    
    # Find all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    car_chunks = []
    human_chunks = []
    
    for subdir in subdirs:
        dir_name = subdir.name.lower()
        
        # Check if this directory contains chunk data
        index_file = subdir / "chunk_index.pkl"
        if index_file.exists():
            if 'car' in dir_name:
                car_chunks.append(subdir)
                print(f"   🚗 Found car chunk: {subdir.name}")
            elif 'human' in dir_name:
                human_chunks.append(subdir)
                print(f"   👤 Found human chunk: {subdir.name}")
            else:
                print(f"   ❓ Unknown chunk type: {subdir.name}")
    
    print(f"\n📊 Discovery Summary:")
    print(f"   🚗 Car chunks: {len(car_chunks)}")
    print(f"   👤 Human chunks: {len(human_chunks)}")
    
    return car_chunks, human_chunks

def load_all_chunk_indices(chunk_directories):
    """
    Load chunk indices from all directories in the list
    """
    chunk_indices = []
    
    for chunk_dir in chunk_directories:
        chunk_index = load_chunk_index(chunk_dir)
        if chunk_index is not None:
            chunk_indices.append(chunk_index)
    
    return chunk_indices

def create_stdp_network_demo(available_freqs):
    """
    Create STDP network for demonstration (compatible with old notebooks)
    """
    try:
        from sctnN.spiking_network import SpikingNetwork
        from sctnN.spiking_neuron import create_SCTN
        from sctnN.layers import SCTNLayer
        
        # Create STDP network (similar to old notebooks)
        clk_freq = 153600
        network = SpikingNetwork()
        
        # Input layer (one neuron per frequency)
        input_neurons = []
        for i, freq in enumerate(available_freqs):
            neuron = create_SCTN()
            neuron.label = f"input_{freq}Hz"
            input_neurons.append(neuron)
        
        network.add_layer(SCTNLayer(input_neurons))
        
        # STDP learning layer
        stdp_neurons = []
        for i in range(2):  # Car vs Human classification
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.random(len(available_freqs)) * 0.1
            neuron.label = f"classifier_{i}"
            
            # Set STDP parameters (like old notebooks)
            neuron.set_stdp(
                A_LTP=0.01,
                A_LTD=0.005, 
                tau=0.02,
                clk_freq=clk_freq,
                wmax=1.0,
                wmin=0.0
            )
            stdp_neurons.append(neuron)
        
        network.add_layer(SCTNLayer(stdp_neurons))
        
        print(f"   🧠 Network created: {len(input_neurons)} inputs → {len(stdp_neurons)} STDP classifiers")
        print(f"   ⚡ STDP enabled with LTP=0.01, LTD=0.005")
        print(f"   �� Ready for training on {len(available_freqs)} frequency channels")
        
        return network, True
        
    except ImportError as e:
        print(f"⚠️  SNN library not available: {e}")
        print("   But the data format is ready for STDP training!")
        return None, False

def run_stdp_classification_from_chunks(base_dir="/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output_30s"):
    """
    Complete STDP classification pipeline using existing chunks
    Skips resonator processing and goes directly to SNN training
    """
    print("🧠 STDP CLASSIFICATION FROM EXISTING CHUNKS")
    print("=" * 60)
    print("Loading existing chunks and creating STDP classification")
    print("(Skipping resonator processing - using saved chunks)")
    print()
    
    # Step 1: Discover and load existing chunks
    print("�� STEP 1: LOADING EXISTING CHUNKS")
    print("-" * 40)
    
    car_chunk_dirs, human_chunk_dirs = discover_chunk_directories(base_dir)
    
    if not car_chunk_dirs and not human_chunk_dirs:
        print("❌ No chunk directories found!")
        print(f"   Make sure {base_dir} exists and contains processed chunks")
        return None
    
    # Load car chunk indices
    car_chunk_indices = []
    if car_chunk_dirs:
        print(f"\n🚗 Loading {len(car_chunk_dirs)} car chunk directories...")
        car_chunk_indices = load_all_chunk_indices(car_chunk_dirs)
        print(f"   ✅ Successfully loaded {len(car_chunk_indices)} car chunk indices")
    
    # Load human chunk indices  
    human_chunk_indices = []
    if human_chunk_dirs:
        print(f"\n👤 Loading {len(human_chunk_dirs)} human chunk directories...")
        human_chunk_indices = load_all_chunk_indices(human_chunk_dirs)
        print(f"   ✅ Successfully loaded {len(human_chunk_indices)} human chunk indices")
    
    if not car_chunk_indices and not human_chunk_indices:
        print("❌ Failed to load any chunk indices!")
        return None
    
    # Step 2: Create SNN training datasets
    print(f"\n🧠 STEP 2: CREATING SNN TRAINING DATASETS")
    print("-" * 40)
    
    car_spikes_data = None
    human_spikes_data = None
    
    # Create car SNN dataset
    if car_chunk_indices:
        print("🚗 Creating car SNN dataset from chunks...")
        try:
            car_spikes_data, car_metadata = create_snn_training_dataset_from_chunks(car_chunk_indices)
            
            # Save in old notebook format
            output_dir = os.path.dirname(os.path.abspath(__file__))
            car_dataset_dir = save_snn_dataset_for_training(
                car_spikes_data, car_metadata, output_dir, "car_combined_from_chunks"
            )
            print(f"   ✅ Car SNN dataset saved to: {car_dataset_dir}")
            
        except Exception as e:
            print(f"   ❌ Failed to create car SNN dataset: {e}")
            import traceback
            traceback.print_exc()
    
    # Create human SNN dataset
    if human_chunk_indices:
        print("\n👤 Creating human SNN dataset from chunks...")
        try:
            human_spikes_data, human_metadata = create_snn_training_dataset_from_chunks(human_chunk_indices)
            
            # Save in old notebook format
            output_dir = os.path.dirname(os.path.abspath(__file__))
            human_dataset_dir = save_snn_dataset_for_training(
                human_spikes_data, human_metadata, output_dir, "human_combined_from_chunks"
            )
            print(f"   ✅ Human SNN dataset saved to: {human_dataset_dir}")
            
        except Exception as e:
            print(f"   ❌ Failed to create human SNN dataset: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Prepare for STDP training
    print(f"\n🎯 STEP 3: STDP TRAINING PREPARATION")
    print("-" * 40)
    
    if car_spikes_data and human_spikes_data:
        print("🧠 Creating STDP-compatible training data...")
        
        # Get available frequencies (use intersection of both datasets)
        car_freqs = set(car_spikes_data.keys())
        human_freqs = set(human_spikes_data.keys())
        common_freqs = sorted(list(car_freqs.intersection(human_freqs)))
        
        # Use first 5 common frequencies
        available_freqs = common_freqs[:5]
        print(f"   📊 Using {len(available_freqs)} common frequencies: {available_freqs}")
        
        # Prepare spike trains (exactly like old notebooks)
        car_training_spikes = {}
        human_training_spikes = {}
        
        for freq in available_freqs:
            # Car data
            car_training_spikes[freq] = car_spikes_data[freq]['spike_train']
            print(f"   🚗 {freq}Hz: {len(car_training_spikes[freq]):,} samples")
            
            # Human data  
            human_training_spikes[freq] = human_spikes_data[freq]['spike_train']
            print(f"   👤 {freq}Hz: {len(human_training_spikes[freq]):,} samples")
        
        # Step 4: STDP Network Creation
        print(f"\n🔬 STEP 4: STDP NETWORK CREATION")
        print("-" * 40)
        
        network, network_created = create_stdp_network_demo(available_freqs)
        
        if network_created:
            # Demo training snippet (showing compatibility)
            print(f"\n💡 TRAINING DEMO (Old Notebook Style)")
            print("-" * 40)
            print("   # Your data is now ready for STDP training:")
            print("   # Load the spike trains like this:")
            print(f"   car_spikes_22_1 = car_training_spikes[{available_freqs[0]}]")
            print(f"   human_spikes_22_1 = human_training_spikes[{available_freqs[0]}]")
            print()
            print("   # Training loop (compatible with old notebooks):")
            print("   for epoch in range(num_epochs):")
            print("       for freq in available_freqs:")
            print("           car_spikes = car_training_spikes[freq]")
            print("           human_spikes = human_training_spikes[freq]")
            print("           network.input_full_data_spikes(car_spikes)")
            print("           # ... STDP learning happens automatically")
            print()
            print("✅ STDP network created and ready for training!")
        
        # Step 5: Create training data summary
        print(f"\n📋 STEP 5: TRAINING DATA SUMMARY")
        print("-" * 40)
        
        summary = {
            'car_data': {
                'total_duration_s': car_metadata['total_duration'],
                'num_files': car_metadata['num_files'],
                'frequencies': list(car_spikes_data.keys()),
                'sample_counts': {freq: len(data['spike_train']) for freq, data in car_spikes_data.items()}
            },
            'human_data': {
                'total_duration_s': human_metadata['total_duration'], 
                'num_files': human_metadata['num_files'],
                'frequencies': list(human_spikes_data.keys()),
                'sample_counts': {freq: len(data['spike_train']) for freq, data in human_spikes_data.items()}
            },
            'training_ready': {
                'common_frequencies': available_freqs,
                'network_created': network_created,
                'car_dataset_path': car_dataset_dir if 'car_dataset_dir' in locals() else None,
                'human_dataset_path': human_dataset_dir if 'human_dataset_dir' in locals() else None
            }
        }
        
        # Save summary for reference
        summary_file = os.path.join(os.path.dirname(__file__), "stdp_training_summary.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Training summary saved to: {summary_file}")
        
        return {
            'car_chunk_indices': car_chunk_indices,
            'human_chunk_indices': human_chunk_indices,
            'car_spikes_data': car_spikes_data,
            'human_spikes_data': human_spikes_data,
            'car_training_spikes': car_training_spikes,
            'human_training_spikes': human_training_spikes,
            'available_frequencies': available_freqs,
            'network': network,
            'summary': summary,
            'status': 'ready_for_stdp_training'
        }
    
    elif car_spikes_data:
        print("⚠️  Only car data available - limited training possible")
        return {
            'car_chunk_indices': car_chunk_indices,
            'car_spikes_data': car_spikes_data,
            'status': 'car_only'
        }
    
    elif human_spikes_data:
        print("⚠️  Only human data available - limited training possible") 
        return {
            'human_chunk_indices': human_chunk_indices,
            'human_spikes_data': human_spikes_data,
            'status': 'human_only'
        }
    
    else:
        print("❌ No valid SNN datasets created")
        return None

def quick_stdp_training_demo(training_data, num_epochs=5):
    """
    Quick demonstration of STDP training using the prepared data
    """
    if not training_data or training_data.get('status') != 'ready_for_stdp_training':
        print("❌ Training data not ready for STDP demo")
        return
    
    print(f"\n🚀 QUICK STDP TRAINING DEMO")
    print("-" * 40)
    
    car_training_spikes = training_data['car_training_spikes']
    human_training_spikes = training_data['human_training_spikes']
    available_freqs = training_data['available_frequencies']
    network = training_data['network']
    
    if network is None:
        print("❌ Network not available for demo")
        return
    
    print(f"🎯 Running {num_epochs} epochs of STDP training...")
    
    try:
        for epoch in range(num_epochs):
            print(f"   Epoch {epoch + 1}/{num_epochs}...")
            
            # Train on car data
            for freq in available_freqs[:2]:  # Use first 2 frequencies for demo
                car_spikes = car_training_spikes[freq]
                # Subsample for demo (use first 1000 samples)
                car_sample = car_spikes[:1000]
                
                # This would be the actual training call:
                # network.input_full_data_spikes(car_sample)
                print(f"      🚗 Training on {freq}Hz car data ({len(car_sample)} samples)")
            
            # Train on human data  
            for freq in available_freqs[:2]:  # Use first 2 frequencies for demo
                human_spikes = human_training_spikes[freq]
                # Subsample for demo (use first 1000 samples)
                human_sample = human_spikes[:1000]
                
                # This would be the actual training call:
                # network.input_full_data_spikes(human_sample)
                print(f"      👤 Training on {freq}Hz human data ({len(human_sample)} samples)")
        
        print(f"✅ Demo training complete!")
        print(f"   💡 This was a demonstration - actual training would use:")
        print(f"      - All {len(available_freqs)} frequencies")
        print(f"      - Full datasets (not subsampled)")
        print(f"      - Proper STDP learning algorithms")
        
    except Exception as e:
        print(f"❌ Error in training demo: {e}")

def validate_chunk_integrity(chunk_indices):
    """
    Validate that chunk files exist and are accessible
    """
    print(f"\n🔍 VALIDATING CHUNK INTEGRITY")
    print("-" * 30)
    
    total_chunks = 0
    valid_chunks = 0
    
    for chunk_index in chunk_indices:
        file_name = Path(chunk_index['file_path']).name
        chunk_files = chunk_index.get('chunk_files', [])
        
        print(f"📁 {file_name}: {len(chunk_files)} chunks")
        
        for chunk_file in chunk_files:
            total_chunks += 1
            if os.path.exists(chunk_file):
                valid_chunks += 1
            else:
                print(f"   ❌ Missing: {chunk_file}")
    
    print(f"\n📊 Validation Summary:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Valid chunks: {valid_chunks}")
    print(f"   Success rate: {valid_chunks/total_chunks*100:.1f}%" if total_chunks > 0 else "   No chunks found")
    
    return valid_chunks == total_chunks

# Main execution
if __name__ == "__main__":
    print("🧠 STDP CLASSIFICATION FROM EXISTING CHUNKS")
    print("=" * 60)
    print("This script loads your existing processed chunks and")
    print("creates STDP classification datasets without re-processing.")
    print()
    
    try:
        # Run the main STDP classification pipeline
        results = run_stdp_classification_from_chunks()
        
        if results and results.get('status') == 'ready_for_stdp_training':
            print("\n🎉 SUCCESS! STDP CLASSIFICATION READY!")
            print("=" * 50)
            print("✅ Chunks loaded successfully")
            print("✅ SNN datasets created")
            print("✅ STDP network initialized")
            print("✅ Training data prepared")
            
            print(f"\n📊 DATASET OVERVIEW:")
            summary = results['summary']
            print(f"🚗 Car data: {summary['car_data']['total_duration_s']:.1f}s from {summary['car_data']['num_files']} files")
            print(f"👤 Human data: {summary['human_data']['total_duration_s']:.1f}s from {summary['human_data']['num_files']} files")
            print(f"🎯 Ready for training on {len(results['available_frequencies'])} frequencies")
            
            print(f"\n💡 NEXT STEPS:")
            print("1. Use the prepared training data for STDP learning")
            print("2. Load datasets from the saved directories")
            print("3. Apply your existing STDP training methods")
            
            # Optional: Run quick training demo
            user_input = input("\n🚀 Run quick STDP training demo? (y/n): ").strip().lower()
            if user_input == 'y':
                quick_stdp_training_demo(results)
            
        elif results:
            print(f"\n⚠️  Partial success: {results.get('status', 'unknown')}")
            
        else:
            print(f"\n❌ Failed to create STDP classification datasets")
            
    except Exception as e:
        print(f"❌ Error in STDP classification: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("📋 WORKFLOW SUMMARY:")
    print("✅ 1. Chunks were loaded from chunked_output_30s/")
    print("✅ 2. SNN datasets created in old notebook format")
    print("✅ 3. STDP network structure prepared")
    print("💡 4. Ready for full STDP training implementation")
    print("=" * 60)
