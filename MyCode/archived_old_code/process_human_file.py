#!/usr/bin/env python3
"""
Human Stepping File Processor for Resonator-Based Spikegram Generation
=====================================================================

This script processes a 60-second geophone file containing:
- 32.5 seconds of human stepping (0-32.5s)
- 27.5 seconds of background/no stepping (32.5-60s)

The file is split into these two categories and processed using the 
resonator-based pipeline to generate spikegrams for each segment.

Output directories:
- /chunked_output/human_new/
- /chunked_output/human_nothing_new/
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import all necessary functions from the main file
import sys
sctn_library_path = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_library_path)

from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator
from sctnN.spiking_neuron import create_SCTN, BINARY

# Import processing functions (these should be available from the main file)
from Geophone_Model_SNN import (
    normalize_signal,
    resample_signal,
    get_resonator_grid,
    process_with_resonator_grid_parallel,
    events_to_max_spectrogram,
    spikes_to_bands,
    create_chunk_visualization,
    compute_fft_spectrogram,
    bands
)

# Configuration
CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
SAMPLING_FREQ = 1000  # 1000 Hz
HUMAN_DURATION = 32.5  # seconds
TOTAL_DURATION = 60.0  # seconds

def load_and_split_human_file(file_path, human_end_time=32.5, sampling_freq=1000):
    """
    Load the 60-second geophone file and split it into human and background sections.
    
    Args:
        file_path (str): Path to the CSV file
        human_end_time (float): Time in seconds where human stepping ends
        sampling_freq (int): Sampling frequency in Hz
        
    Returns:
        tuple: (human_signal, background_signal, time_human, time_background)
    """
    print(f"📂 Loading and splitting file: {file_path}")
    
    try:
        # Load the full file
        data = pd.read_csv(file_path)
        
        # Extract signal data
        if 'amplitude' in data.columns:
            full_signal = data['amplitude'].values
        else:
            full_signal = data.iloc[:, 1].values
        
        print(f"   📊 Loaded {len(full_signal)} samples ({len(full_signal)/sampling_freq:.1f} seconds)")
        
        # Calculate split point
        split_sample = int(human_end_time * sampling_freq)
        
        # Split the signal
        human_signal = full_signal[:split_sample]
        background_signal = full_signal[split_sample:]
        
        # Normalize both signals
        human_signal = normalize_signal(human_signal)
        background_signal = normalize_signal(background_signal)
        
        # Create time axes
        time_human = np.arange(len(human_signal)) / sampling_freq
        time_background = np.arange(len(background_signal)) / sampling_freq
        
        print(f"   👤 Human section: {len(human_signal)} samples ({len(human_signal)/sampling_freq:.1f}s)")
        print(f"   🔇 Background section: {len(background_signal)} samples ({len(background_signal)/sampling_freq:.1f}s)")
        
        return human_signal, background_signal, time_human, time_background
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None, None

def process_signal_section(signal, time_axis, section_name, output_dir, chunk_duration=30, num_processes=15):
    """
    Process a signal section using the resonator pipeline and save results.
    
    Args:
        signal (np.ndarray): Signal data
        time_axis (np.ndarray): Time axis
        section_name (str): Name of the section ('human_new' or 'human_nothing_new')
        output_dir (str): Output directory path
        chunk_duration (float): Duration of each chunk in seconds
        num_processes (int): Number of parallel processes
        
    Returns:
        list: List of processed chunk results
    """
    print(f"\n🔄 Processing {section_name} section...")
    print("=" * 60)
    
    # Create output directory for this section
    section_output_dir = os.path.join(output_dir, section_name)
    os.makedirs(section_output_dir, exist_ok=True)
    
    # Calculate duration and chunks
    total_duration = len(signal) / SAMPLING_FREQ
    print(f"📊 Section duration: {total_duration:.1f} seconds")
    
    # Determine chunk boundaries
    chunk_boundaries = []
    current_pos = 0
    chunk_idx = 0
    
    while current_pos < total_duration:
        next_pos = min(current_pos + chunk_duration, total_duration)
        chunk_boundaries.append((current_pos, next_pos))
        current_pos = next_pos
        
        if current_pos >= total_duration:
            break
    
    print(f"📂 Will create {len(chunk_boundaries)} chunks")
    
    # Auto-detect resonator grid (use human-optimized for human section)
    if 'human' in section_name:
        clk_resonators = {
            153600: [
                22.1,  # LOW_FREQ
                30.5, 33.9, 34.7, 41.2,  # Reduced CAR coverage
                50.9, 52.6,  # Enhanced MID_GAP
                76.3, 63.6,  # ALL HUMAN coverage
                95.4  # Minimal HIGH_FREQ
            ]
        }
        print("🧠 Using human-optimized resonator grid")
    else:
        clk_resonators = {
            153600: [
                22.1, 28.8,  # LOW_FREQ
                30.5, 34.7, 37.2, 40.2, 43.6, 47.7,  # Enhanced CAR coverage
                52.6, 58.7,  # MID_GAP
                63.6, 69.4, 76.3,  # Reduced HUMAN
                89.8, 95.4  # HIGH_FREQ
            ]
        }
        print("🔇 Using background-optimized resonator grid")
    
    chunk_results = []
    
    # Process each chunk
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        current_chunk_duration = chunk_end - chunk_start
        
        if current_chunk_duration <= 0:
            break
        
        print(f"\n--- Processing chunk {chunk_idx}: {chunk_start:.1f}s-{chunk_end:.1f}s ---")
        
        # Extract chunk from signal
        start_sample = int(chunk_start * SAMPLING_FREQ)
        end_sample = int(chunk_end * SAMPLING_FREQ)
        chunk_signal = signal[start_sample:end_sample]
        chunk_time = time_axis[start_sample:end_sample]
        
        if len(chunk_signal) == 0:
            print(f"⚠️  Skipping empty chunk {chunk_idx}")
            continue
        
        try:
            # Process with resonator grid
            print(f"⚡ Processing chunk {chunk_idx} with resonator grid...")
            resonator_output = process_with_resonator_grid_parallel(
                chunk_signal,
                SAMPLING_FREQ,
                clk_resonators,
                current_chunk_duration,
                num_processes=num_processes
            )
            
            # Create spike spectrograms
            print(f"📊 Creating spike spectrograms for chunk {chunk_idx}...")
            max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
                resonator_output,
                current_chunk_duration,
                clk_resonators,
                section_name  # Use section name for adaptive processing
            )
            
            # Group by frequency bands
            spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)
            
            # Prepare chunk results
            chunk_result = {
                'chunk_idx': chunk_idx,
                'start_time': chunk_start,
                'duration': current_chunk_duration,
                'signal': chunk_signal,
                'time': chunk_time,
                'resonator_outputs': resonator_output,
                'max_spikes_spectrogram': max_spikes_spectrogram,
                'spikes_bands_spectrogram': spikes_bands_spectrogram,
                'all_freqs': all_freqs,
                'section_name': section_name
            }
            
            # Create chunk-specific output directory
            chunk_output_dir = os.path.join(section_output_dir, f"chunk_{chunk_idx}")
            os.makedirs(chunk_output_dir, exist_ok=True)
            
            # Save chunk data
            chunk_file = os.path.join(chunk_output_dir, f"chunk_{chunk_idx}_data.pkl")
            with open(chunk_file, 'wb') as f:
                pickle.dump(chunk_result, f)
            
            # Create visualization
            create_enhanced_chunk_visualization(chunk_result, chunk_output_dir)
            
            chunk_results.append(chunk_result)
            print(f"✅ Chunk {chunk_idx} processed and saved")
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_idx}: {e}")
            continue
    
    print(f"\n✅ {section_name} processing complete: {len(chunk_results)} chunks saved")
    
    # Save section index
    section_index = {
        'section_name': section_name,
        'total_duration': total_duration,
        'chunk_duration': chunk_duration,
        'num_chunks': len(chunk_results),
        'chunk_boundaries': chunk_boundaries,
        'chunk_files': [os.path.join(section_output_dir, f"chunk_{i}", f"chunk_{i}_data.pkl") 
                       for i in range(len(chunk_results))]
    }
    
    index_file = os.path.join(section_output_dir, "section_index.pkl")
    with open(index_file, 'wb') as f:
        pickle.dump(section_index, f)
    
    print(f"📋 Section index saved to {index_file}")
    
    return chunk_results

def create_enhanced_chunk_visualization(chunk_results, output_dir):
    """
    Create enhanced visualization for a chunk with FFT spectrogram and spikegram.
    """
    try:
        signal = chunk_results['signal']
        time = chunk_results['time']
        spikes_bands_spectrogram = chunk_results['spikes_bands_spectrogram']
        duration = chunk_results['duration']
        chunk_idx = chunk_results['chunk_idx']
        section_name = chunk_results['section_name']
        
        # Create comprehensive visualization
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1.5, 1.5]})
        
        # Plot 1: Raw Signal
        axs[0].plot(time, signal)
        axs[0].set_title(f'{section_name} - Chunk {chunk_idx} - Raw Signal ({duration:.1f}s)', fontsize=14)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: FFT Spectrogram
        f, t, Sxx = compute_fft_spectrogram(signal, SAMPLING_FREQ, fmin=1, fmax=100, plot=False)
        
        # Create band labels
        band_labels = [f'{fmin}-{fmax} ({band})' for band, (fmin, fmax) in bands.items()]
        
        # Convert FFT spectrogram to band-based representation
        fft_bin_spectogram = np.zeros((len(bands), len(t)))
        for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            f_indices = np.where((f >= fmin) & (f < fmax))[0]
            if len(f_indices) > 0:
                fft_bin_spectogram[i] = np.mean(Sxx[f_indices], axis=0)
        
        # Apply log transformation
        fft_bin_spectogram = 10 * np.log10(fft_bin_spectogram + 1e-10)
        
        im1 = axs[1].imshow(fft_bin_spectogram, aspect='auto', cmap='jet', origin='lower',
                   extent=[0, duration, 0, len(bands)])
        axs[1].set_yticks(np.arange(len(band_labels)) + 0.5)
        axs[1].set_yticklabels(band_labels)
        axs[1].set_title(f'{section_name} - Chunk {chunk_idx} - FFT Spectrogram', fontsize=14)
        axs[1].set_ylabel('Frequency Band')
        fig.colorbar(im1, ax=axs[1], label='Power (dB)', pad=0.01)
        
        # Plot 3: Resonator Spikegram
        target_time_bins = len(t)
        if spikes_bands_spectrogram.shape[1] > target_time_bins:
            factor = spikes_bands_spectrogram.shape[1] // target_time_bins
            if factor > 1:
                reshaped = np.zeros((spikes_bands_spectrogram.shape[0], target_time_bins))
                for i in range(target_time_bins):
                    start_idx = i * factor
                    end_idx = min((i + 1) * factor, spikes_bands_spectrogram.shape[1])
                    if end_idx > start_idx:
                        reshaped[:, i] = np.max(spikes_bands_spectrogram[:, start_idx:end_idx], axis=1)
                spikes_bands_spectrogram = reshaped
        
        # Adaptive visualization parameters
        if 'human' in section_name and 'nothing' not in section_name:
            vmax = np.percentile(spikes_bands_spectrogram, 97)  # Human stepping
        else:
            vmax = np.percentile(spikes_bands_spectrogram, 99)  # Background
        
        im2 = axs[2].imshow(spikes_bands_spectrogram, aspect='auto', cmap='jet', origin='lower',
                          extent=[0, duration, 0, len(bands)], vmin=0, vmax=vmax)
        axs[2].set_yticks(np.arange(len(band_labels)) + 0.5)
        axs[2].set_yticklabels(band_labels)
        axs[2].set_title(f'{section_name} - Chunk {chunk_idx} - Resonator Spikegram', fontsize=14)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Frequency Band')
        fig.colorbar(im2, ax=axs[2], label='Spike Activity', pad=0.01)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"{section_name}_chunk_{chunk_idx}_visualization.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {plot_file}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def process_human_stepping_file(file_path, human_end_time=32.5, chunk_duration=30, num_processes=15):
    """
    Main function to process the human stepping file.
    
    Args:
        file_path (str): Path to the 60-second geophone CSV file
        human_end_time (float): Time in seconds where human stepping ends (default: 32.5)
        chunk_duration (float): Duration of each processing chunk in seconds
        num_processes (int): Number of parallel processes for resonator processing
        
    Returns:
        dict: Summary of processing results
    """
    print("🧠 HUMAN STEPPING FILE PROCESSOR")
    print("=" * 80)
    print(f"📂 Input file: {file_path}")
    print(f"👤 Human stepping: 0 - {human_end_time}s")
    print(f"🔇 Background: {human_end_time}s - {TOTAL_DURATION}s")
    print(f"📁 Output directory: {CHUNKED_OUTPUT_DIR}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Validate input file
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    # Load and split the file
    human_signal, background_signal, time_human, time_background = load_and_split_human_file(
        file_path, human_end_time, SAMPLING_FREQ
    )
    
    if human_signal is None:
        print("❌ Error: Failed to load and split the file")
        return None
    
    # Create main output directory
    os.makedirs(CHUNKED_OUTPUT_DIR, exist_ok=True)
    
    # Process human stepping section
    print(f"\n🔄 PROCESSING HUMAN STEPPING SECTION")
    human_results = process_signal_section(
        human_signal, time_human, "human_new", CHUNKED_OUTPUT_DIR, 
        chunk_duration, num_processes
    )
    
    # Process background section
    print(f"\n🔄 PROCESSING BACKGROUND SECTION")
    background_results = process_signal_section(
        background_signal, time_background, "human_nothing_new", CHUNKED_OUTPUT_DIR,
        chunk_duration, num_processes
    )
    
    # Calculate processing time
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n✅ PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"👤 Human chunks processed: {len(human_results)}")
    print(f"🔇 Background chunks processed: {len(background_results)}")
    print(f"📁 Human output: {CHUNKED_OUTPUT_DIR}/human_new/")
    print(f"📁 Background output: {CHUNKED_OUTPUT_DIR}/human_nothing_new/")
    
    # Create overall summary
    summary = {
        'input_file': file_path,
        'human_end_time': human_end_time,
        'total_duration': TOTAL_DURATION,
        'processing_time': total_time,
        'human_chunks': len(human_results),
        'background_chunks': len(background_results),
        'output_directories': {
            'human': os.path.join(CHUNKED_OUTPUT_DIR, "human_new"),
            'background': os.path.join(CHUNKED_OUTPUT_DIR, "human_nothing_new")
        },
        'chunk_duration': chunk_duration,
        'sampling_frequency': SAMPLING_FREQ
    }
    
    # Save overall summary
    summary_file = os.path.join(CHUNKED_OUTPUT_DIR, "processing_summary.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"📋 Processing summary saved: {summary_file}")
    
    return summary

# Example usage
if __name__ == "__main__":
    # Example file path - update this to your actual file path
    file_path = "/path/to/your/60second_geophone_file.csv"
    
    print("🧠 HUMAN STEPPING RESONATOR PROCESSOR")
    print("=" * 80)
    print("This script processes a 60-second geophone file containing:")
    print("  • 32.5 seconds of human stepping (0-32.5s)")
    print("  • 27.5 seconds of background/no stepping (32.5-60s)")
    print()
    print("The file is split and processed using resonator-based spikegram generation.")
    print("=" * 80)
    
    # Check if file path needs to be provided
    if not os.path.exists(file_path) or "/path/to/" in file_path:
        print("❌ Please update the file_path variable with your actual file path")
        print("Example usage:")
        print('    file_path = "/home/user/data/geophone_60sec.csv"')
        print('    summary = process_human_stepping_file(file_path)')
    else:
        # Process the file
        summary = process_human_stepping_file(
            file_path=file_path,
            human_end_time=32.5,  # Human stepping ends at 32.5 seconds
            chunk_duration=30,    # Process in 30-second chunks
            num_processes=15      # Use 15 parallel processes
        )
        
        if summary:
            print("\n🎉 SUCCESS! File processed successfully.")
            print(f"Check the output directories:")
            print(f"  👤 Human: {summary['output_directories']['human']}")
            print(f"  🔇 Background: {summary['output_directories']['background']}") 