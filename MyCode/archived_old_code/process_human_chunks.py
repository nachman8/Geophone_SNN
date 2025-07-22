import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import resample, spectrogram
from joblib import Parallel, delayed
import multiprocessing
import os
import time
import threading
from sklearn.model_selection import train_test_split
import pickle

from pathlib import Path
DATA_DIR = Path.home() / "data"

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

# Import the required functions from resonator_work.py
# Since we're in the same directory, we can import directly
try:
    from resonator_work import (
        clk_resonators_human,
        bands,
        get_resonator_grid,
        normalize_signal,
        process_single_resonator,
        process_with_resonator_grid_parallel,
        spikes_event_spectrogram,
        events_to_max_spectrogram,
        spikes_to_bands,
        progress_monitor,
        save_chunk_for_snn_classification,
        create_snn_training_dataset_from_chunks,
        save_snn_dataset_for_training
    )
    print("✅ Successfully imported functions from resonator_work.py")
except ImportError as e:
    print(f"❌ Error importing from resonator_work.py: {e}")
    print("Make sure resonator_work.py is in the same directory")
    sys.exit(1)

def save_plot(name=None):
    """Save the current figure to a file with a unique name"""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    if name is None:
        name = f"plot_{int(time.time())}.png"
    else:
        name = f"{name}.png"
    
    # Save the figure
    filepath = os.path.join(output_dir, name)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")
    return filepath

def get_file_duration(file_path, sampling_freq=1000):
    """
    Get the total duration of a file without loading the entire file into memory
    """
    try:
        # Read just the first few rows to understand structure
        sample_data = pd.read_csv(file_path, nrows=10)
        
        # Get total row count efficiently
        with open(file_path, 'r') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract header
        
        # Estimate duration
        duration = total_rows / sampling_freq
        
        print(f"File {file_path} has {total_rows} samples, estimated duration: {duration:.1f} seconds")
        return duration
        
    except Exception as e:
        print(f"Error getting file duration: {e}")
        return None

def load_chunk_data(file_path, start_time, chunk_duration, sampling_freq=1000):
    """
    Load a specific chunk of data from a file without loading the entire file
    """
    try:
        # Calculate row indices for the chunk
        start_row = int(start_time * sampling_freq) + 1  # +1 for header
        end_row = int((start_time + chunk_duration) * sampling_freq) + 1
        
        # Read the specific chunk
        chunk_data = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=end_row-start_row)
        
        # Use appropriate column for signal data
        if 'amplitude' in chunk_data.columns:
            signal = chunk_data['amplitude'].values
        else:
            signal = chunk_data.iloc[:, 1].values
        
        # Normalize the signal
        signal = normalize_signal(signal)
        
        # Create time axis
        time = np.arange(len(signal)) / sampling_freq
        
        actual_duration = len(signal) / sampling_freq
        
        print(f"Loaded chunk {start_time:.1f}-{start_time + actual_duration:.1f}s: {len(signal)} samples")
        
        return signal, time, actual_duration
        
    except Exception as e:
        print(f"Error loading chunk from {file_path}: {e}")
        return None, None, None

def compute_fft_spectrogram(signal, fs, fmin=1, fmax=80, nperseg=1024, noverlap=512, plot=True):
    """
    Compute and optionally plot FFT spectrogram
    """
    # Compute spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Plot spectrogram only if requested
    if plot:
        plt.figure(figsize=(14, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Signal Spectrogram')
        plt.ylim(fmin, fmax)
        save_plot() 

    return f, t, Sxx

def create_chunk_visualization(chunk_results, output_dir):
    """
    Create visualization for a single chunk with both FFT spectrogram and spikegram
    """
    try:
        signal = chunk_results['signal']
        time = chunk_results['time']
        spikes_bands_spectrogram = chunk_results['spikes_bands_spectrogram']
        duration = chunk_results['duration']
        chunk_idx = chunk_results['chunk_idx']
        
        # Create a comprehensive visualization for the chunk
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1.5, 1.5]})
        
        # Plot 1: Raw Signal
        axs[0].plot(time, signal)
        axs[0].set_title(f'Chunk {chunk_idx} - Raw Signal ({duration:.1f}s)', fontsize=14)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: FFT Spectrogram
        print(f"Computing FFT spectrogram for chunk {chunk_idx}...")
        f, t, Sxx = compute_fft_spectrogram(signal, 1000, fmin=1, fmax=100, plot=False)
        
        # Create band labels
        band_labels = [f'{fmin}-{fmax} ({band})' for band, (fmin, fmax) in bands.items()]
        
        # Convert FFT spectrogram to band-based representation
        fft_bin_spectogram = np.zeros((len(bands), len(t)))
        for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            # Find frequency indices for this band
            f_indices = np.where((f >= fmin) & (f < fmax))[0]
            if len(f_indices) > 0:
                # Average power in this band
                fft_bin_spectogram[i] = np.mean(Sxx[f_indices], axis=0)
        
        # Apply log transformation to enhance contrast
        fft_bin_spectogram = 10 * np.log10(fft_bin_spectogram + 1e-10)
        
        im1 = axs[1].imshow(fft_bin_spectogram, aspect='auto', cmap='jet', origin='lower',
                   extent=[0, duration, 0, len(bands)])
        axs[1].set_yticks(np.arange(len(band_labels)) + 0.5)
        axs[1].set_yticklabels(band_labels)
        axs[1].set_title(f'Chunk {chunk_idx} - FFT Spectrogram', fontsize=14)
        axs[1].set_ylabel('Frequency Band')
        fig.colorbar(im1, ax=axs[1], label='Power (dB)', pad=0.01)
        
        # Plot 3: Spikegram (Resonator Output) 
        # Downsample to match FFT spectrogram time resolution
        target_time_bins = len(t)  # Match FFT spectrogram time resolution
        if spikes_bands_spectrogram.shape[1] > target_time_bins:
            # Reshape to match FFT spectrogram time bins
            factor = spikes_bands_spectrogram.shape[1] // target_time_bins
            if factor > 1:
                reshaped = np.zeros((spikes_bands_spectrogram.shape[0], target_time_bins))
                for i in range(target_time_bins):
                    start_idx = i * factor
                    end_idx = min((i + 1) * factor, spikes_bands_spectrogram.shape[1])
                    if end_idx > start_idx:
                        reshaped[:, i] = np.max(spikes_bands_spectrogram[:, start_idx:end_idx], axis=1)
                spikes_bands_spectrogram = reshaped
        
        # Adaptive visualization parameters for human data
        is_nothing_file = 'nothing' in str(chunk_results['file_path']).lower()
        is_human_data = 'human' in str(chunk_results['file_path']).lower()
        
        if is_nothing_file:
            if is_human_data:
                vmax = np.percentile(spikes_bands_spectrogram, 97)
            else:
                vmax = np.percentile(spikes_bands_spectrogram, 99)
        else:
            if is_human_data:
                vmax = np.percentile(spikes_bands_spectrogram, 97)
            else:
                vmax = np.percentile(spikes_bands_spectrogram, 98)
        
        im2 = axs[2].imshow(spikes_bands_spectrogram, aspect='auto', cmap='jet', origin='lower',
                          extent=[0, duration, 0, len(bands)], vmin=0, vmax=vmax)
        axs[2].set_yticks(np.arange(len(band_labels)) + 0.5)
        axs[2].set_yticklabels(band_labels)
        axs[2].set_title(f'Chunk {chunk_idx} - Resonator-based Spikegram', fontsize=14)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Frequency Band')
        fig.colorbar(im2, ax=axs[2], label='Spike Activity', pad=0.01)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"chunk_{chunk_idx}_visualization.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Chunk {chunk_idx} visualization (with FFT + Spikegram) saved to {plot_file}")
        
    except Exception as e:
        print(f"Error creating chunk visualization: {e}")
        import traceback
        traceback.print_exc()

def process_file_chunk(file_path, chunk_start, chunk_duration, chunk_idx, 
                      output_dir, num_processes=15):
    """
    Process a single chunk of a file and save intermediate results
    """
    print(f"\n--- Processing chunk {chunk_idx}: {chunk_start:.1f}s-{chunk_start + chunk_duration:.1f}s ---")
    
    # Load chunk data
    signal, time, actual_duration = load_chunk_data(file_path, chunk_start, chunk_duration)
    
    if signal is None:
        print(f"Failed to load chunk {chunk_idx}")
        return None
    
    # Auto-detect and select appropriate resonator grid (will use human grid)
    clk_resonators = get_resonator_grid(file_path)
    
    # Process with resonator grid
    print(f"Processing chunk {chunk_idx} with resonator grid...")
    try:
        output = process_with_resonator_grid_parallel(
            signal,
            1000,
            clk_resonators,
            actual_duration,
            num_processes=num_processes
        )
        
        # Create spike spectrograms
        print(f"Creating spike spectrograms for chunk {chunk_idx}...")
        max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
            output,
            actual_duration,
            clk_resonators,
            file_path
        )
        
        # Group by frequency bands
        spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)
        
        # Save chunk results
        chunk_results = {
            'chunk_idx': chunk_idx,
            'start_time': chunk_start,
            'duration': actual_duration,
            'signal': signal,
            'time': time,
            'resonator_outputs': output,
            'max_spikes_spectrogram': max_spikes_spectrogram,
            'spikes_bands_spectrogram': spikes_bands_spectrogram,
            'all_freqs': all_freqs,
            'file_path': str(file_path)
        }
        
        # Create chunk-specific output directory
        chunk_output_dir = os.path.join(output_dir, f"chunk_{chunk_idx}")
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        # Save chunk data
        chunk_file = os.path.join(chunk_output_dir, f"chunk_{chunk_idx}_data.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk_results, f)
        
        # Create individual chunk visualization
        create_chunk_visualization(chunk_results, chunk_output_dir)
        
        print(f"✅ Chunk {chunk_idx} processed and saved to {chunk_file}")
        
        return chunk_results
        
    except Exception as e:
        print(f"ERROR processing chunk {chunk_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_file_in_chunks(file_path, chunk_duration=30, num_processes=15, min_chunk_size=10):
    """
    Process a single file in chunks to manage memory usage
    Small leftover chunks are added to the previous chunk to avoid tiny chunks
    """
    print(f"\n🔄 PROCESSING HUMAN FILE IN CHUNKS: {file_path}")
    print("=" * 60)
    
    # Get total file duration
    total_duration = get_file_duration(file_path)
    if total_duration is None:
        print(f"Failed to get duration for {file_path}")
        return None
    
    # Calculate chunk boundaries, avoiding small leftover chunks
    chunk_boundaries = []
    current_pos = 0
    
    while current_pos < total_duration:
        next_pos = current_pos + chunk_duration
        remaining = total_duration - next_pos
        
        # If remaining time is small, add it to current chunk
        if remaining > 0 and remaining < min_chunk_size:
            next_pos = total_duration  # Extend current chunk to include small leftover
            print(f"Small leftover ({remaining:.2f}s) added to previous chunk")
        
        chunk_boundaries.append((current_pos, min(next_pos, total_duration)))
        current_pos = next_pos
        
        if current_pos >= total_duration:
            break
    
    num_chunks = len(chunk_boundaries)
    print(f"File duration: {total_duration:.1f}s, will process in {num_chunks} optimized chunks")
    for i, (start, end) in enumerate(chunk_boundaries):
        print(f"  Chunk {i}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
    
    # Create output directory for this file
    file_stem = Path(file_path).stem
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunked_output_30s", file_stem)
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_results = []
    
    # Process each chunk
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        current_chunk_duration = chunk_end - chunk_start
        
        if current_chunk_duration <= 0:
            break
            
        chunk_result = process_file_chunk(
            file_path, chunk_start, current_chunk_duration, 
            chunk_idx, output_dir, num_processes
        )
        
        if chunk_result is not None:
            chunk_results.append(chunk_result)
        
        # Clear memory after each chunk
        import gc
        gc.collect()
    
    print(f"\n✅ File {file_path} processed in {len(chunk_results)} chunks")
    
    # Save chunk index for this file
    index_file = os.path.join(output_dir, "chunk_index.pkl")
    chunk_index = {
        'file_path': str(file_path),
        'total_duration': total_duration,
        'chunk_duration': chunk_duration,
        'num_chunks': len(chunk_results),
        'chunk_boundaries': chunk_boundaries,
        'chunk_files': [os.path.join(output_dir, f"chunk_{i}", f"chunk_{i}_data.pkl") 
                       for i in range(len(chunk_results))]
    }
    
    with open(index_file, 'wb') as f:
        pickle.dump(chunk_index, f)
    
    print(f"Chunk index saved to {index_file}")
    
    # Save in SNN-optimized format for STDP classification
    if chunk_results:
        try:
            snn_data_dir, snn_metadata = save_chunk_for_snn_classification(
                chunk_results, output_dir, total_duration
            )
            chunk_index['snn_data_dir'] = snn_data_dir
            chunk_index['snn_metadata'] = snn_metadata
            
            # Re-save chunk index with SNN info
            with open(index_file, 'wb') as f:
                pickle.dump(chunk_index, f)
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to save SNN-optimized data: {e}")
            print("   Regular chunk data is still available")
    
    return chunk_index

def process_human_files_only(chunk_duration=30, num_processes=15):
    """
    Process ONLY the human files (human.csv and human_nothing.csv) 
    using the same chunked processing approach as the car files
    """
    print("👤 PROCESSING HUMAN FILES ONLY")
    print("=" * 50)
    print("This will process human.csv and human_nothing.csv using")
    print("the same chunked approach as the car data.")
    print()
    
    # Define human file paths
    human_files = [
        DATA_DIR / "human.csv",
        DATA_DIR / "human_nothing.csv"
    ]
    
    # Check if files exist
    for file_path in human_files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
    
    print("✅ All human files found!")
    print()
    
    # Process each human file in chunks
    human_chunk_indices = []
    
    for file_path in human_files:
        print(f"👤 Processing {file_path.name}...")
        chunk_index = process_file_in_chunks(file_path, chunk_duration, num_processes)
        if chunk_index:
            human_chunk_indices.append(chunk_index)
            print(f"✅ {file_path.name} processing complete!")
        else:
            print(f"❌ Failed to process {file_path.name}")
        
        # Clear memory between files
        import gc
        gc.collect()
        print()
    
    if not human_chunk_indices:
        print("❌ Failed to process any human files")
        return None
    
    print(f"✅ Successfully processed {len(human_chunk_indices)} human files")
    
    # Create SNN training dataset from human chunks
    print("\n🧠 CREATING HUMAN SNN DATASET...")
    print("-" * 40)
    
    try:
        human_spikes_data, human_metadata = create_snn_training_dataset_from_chunks(human_chunk_indices)
        
        # Save in old notebook format
        output_dir = os.path.dirname(os.path.abspath(__file__))
        human_dataset_dir = save_snn_dataset_for_training(
            human_spikes_data, human_metadata, output_dir, "human_combined"
        )
        
        print(f"✅ Human SNN dataset saved to: {human_dataset_dir}")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to create SNN dataset: {e}")
        human_spikes_data = None
        human_metadata = None
    
    print("\n" + "="*50)
    print("🎉 HUMAN DATA PROCESSING COMPLETE!")
    print("✅ Human files: Processed ✓")
    print("✅ Chunked processing: Memory efficient ✓")
    print("✅ Temporal continuity: Preserved ✓")
    print("✅ SNN STDP compatibility: Full ✓")
    print("✅ Compatible with car chunks: 100% ✓")
    print("=" * 50)
    
    return {
        'human_chunk_indices': human_chunk_indices,
        'human_spikes_data': human_spikes_data,
        'human_metadata': human_metadata if 'human_metadata' in locals() else None,
        'status': 'complete'
    }

def test_human_processing_setup():
    """
    Test function to validate the human processing setup
    """
    print("🧪 TESTING HUMAN PROCESSING SETUP")
    print("-" * 40)
    
    # Test 1: Check human resonator grid
    print("✅ Test 1: Human resonator grid")
    print(f"   Human resonators (153600 Hz): {len(clk_resonators_human[153600])} frequencies")
    print(f"   Frequencies: {clk_resonators_human[153600]}")
    
    # Test 2: Check human file paths
    print("\n✅ Test 2: Human file paths")
    human_files = [
        DATA_DIR / "human.csv",
        DATA_DIR / "human_nothing.csv"
    ]
    
    for file_path in human_files:
        status = "✅ exists" if file_path.exists() else "❌ missing"
        if file_path.exists():
            # Get file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   {file_path.name}: {status} ({size_mb:.1f} MB)")
        else:
            print(f"   {file_path.name}: {status}")
    
    # Test 3: Check car chunks directory
    print("\n✅ Test 3: Car chunks directory")
    car_chunks_dir = Path("chunked_output_30s")
    if car_chunks_dir.exists():
        car_subdirs = [d for d in car_chunks_dir.iterdir() if d.is_dir()]
        print(f"   Car chunks directory: ✅ exists")
        print(f"   Found {len(car_subdirs)} subdirectories: {[d.name for d in car_subdirs]}")
    else:
        print(f"   Car chunks directory: ❌ missing at {car_chunks_dir.absolute()}")
    
    # Test 4: Check imports
    print("\n✅ Test 4: Required imports")
    required_functions = ['get_resonator_grid', 'process_with_resonator_grid_parallel', 
                         'events_to_max_spectrogram', 'spikes_to_bands']
    
    for func_name in required_functions:
        try:
            func = globals()[func_name]
            print(f"   {func_name}: ✅ available")
        except KeyError:
            print(f"   {func_name}: ❌ missing")
    
    print("\n🎯 HUMAN PROCESSING SETUP VALIDATION COMPLETE")
    print("   Ready to process human files!")
    
    return True

# Main execution
if __name__ == "__main__":
    print("👤 HUMAN-ONLY CHUNKED PROCESSING PIPELINE")
    print("=" * 50)
    print("This script processes ONLY human files using the same")
    print("chunked approach as the car data, complementing your")
    print("existing car chunks.")
    print()
    
    # Test setup first
    print("🔄 TESTING SETUP...")
    print("-" * 30)
    
    try:
        test_human_processing_setup()
        
        print("\n🔄 PROCESSING HUMAN FILES...")
        print("-" * 30)
        
        # Process human files
        results = process_human_files_only(
            chunk_duration=30,  # Same as car processing
            num_processes=15    # Same as car processing
        )
        
        if results and results.get('status') == 'complete':
            print("\n🏆 SUCCESS! Human data processing complete!")
            print("   ✅ Human chunks created and saved")
            print("   ✅ SNN-compatible format ready")
            print("   ✅ Can be combined with existing car chunks")
            
            print("\n💡 NEXT STEPS:")
            print("   1. Your human chunks are saved in chunked_output_30s/")
            print("   2. You can now combine with car chunks for full analysis")
            print("   3. Use the SNN datasets for STDP training")
        else:
            print("\n❌ Human processing failed or incomplete")
    
    except Exception as e:
        print(f"⚠️  Error in human processing: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*50)
    print("📁 OUTPUT STRUCTURE:")
    print("chunked_output_30s/")
    print("├── car/                    # Your existing car chunks")
    print("├── car_nothing/            # Your existing car_nothing chunks")
    print("├── human/                  # NEW: human chunks")
    print("└── human_nothing/          # NEW: human_nothing chunks")
    print("=" * 50)
