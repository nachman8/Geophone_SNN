#!/usr/bin/env python3
"""
Simple Human Stepping File Processor for Resonator-Based Spikegram Generation
============================================================================

This script processes a 60-second geophone file containing:
- 32.5 seconds of human stepping (0-32.5s)  
- 27.5 seconds of background/no stepping (32.5-60s)

The file is split and processed using resonator-based pipeline to generate spikegrams.

Usage:
    python process_human_file_simple.py /path/to/your/geophone_file.csv
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
import threading
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
from scipy.signal import resample, spectrogram

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add SCTN library path
sctn_library_path = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_library_path)

from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator
from sctnN.spiking_neuron import create_SCTN, BINARY

# Configuration
CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
SAMPLING_FREQ = 1000  # 1000 Hz
HUMAN_DURATION = 32.5  # seconds
TOTAL_DURATION = 60.0  # seconds

# Frequency bands for analysis
bands = {
    'LOW_FREQ': (20, 30),
    'CAR_APPROACH': (30, 34), 
    'CAR_PEAK': (34, 40),
    'CAR_TAIL': (40, 48),
    'MID_GAP': (48, 60),
    'HUMAN_PEAK': (60, 70),
    'HUMAN_TAIL': (70, 80),
    'HIGH_FREQ': (90, 100)
}

# Resonator grids
clk_resonators_human = {
    153600: [
        22.1,  # LOW_FREQ coverage
        30.5, 33.9, 34.7, 41.2,  # Reduced CAR coverage
        50.9, 52.6,  # Enhanced MID_GAP coverage
        76.3, 63.6,  # ALL available HUMAN coverage
        95.4  # Minimal HIGH_FREQ coverage
    ]
}

clk_resonators_background = {
    153600: [
        22.1, 28.8,  # LOW_FREQ coverage
        30.5, 34.7, 37.2, 40.2, 43.6, 47.7,  # Enhanced CAR coverage
        52.6, 58.7,  # MID_GAP coverage
        63.6, 69.4, 76.3,  # Reduced HUMAN coverage
        89.8, 95.4  # HIGH_FREQ coverage
    ]
}

def normalize_signal(signal):
    """Normalize signal to [-1, 1] range"""
    signal_min, signal_max = np.min(signal), np.max(signal)
    if signal_max > signal_min:
        return 2 * (signal - signal_min) / (signal_max - signal_min) - 1
    return np.zeros_like(signal)

def resample_signal(f_new, f_source, data):
    """Resample signal to match a new frequency"""
    n_samples_orig = data.shape[0]
    n_samples_new = int(n_samples_orig * f_new / f_source)
    return resample(data, n_samples_new)

def process_single_resonator(f0, clk_freq, resampled_signal, progress_dict=None, resonator_id=None):
    """Process a single resonator with progress tracking"""
    try:
        # Get closest resonator function
        resonator_func, actual_freq = get_closest_resonator(f0)
        
        # Create resonator
        my_resonator = resonator_func()
        my_resonator.log_out_spikes(-1)
        
        # Progress callback function
        def progress_callback(current, total):
            if progress_dict is not None and resonator_id is not None:
                progress_dict[resonator_id] = (current, total)
        
        # Process signal with progress tracking
        my_resonator.input_full_data(resampled_signal, progress_callback=progress_callback)
        
        # Get output spikes
        output_spikes = my_resonator.neurons[-1].out_spikes()
        
        return output_spikes
        
    except Exception as e:
        print(f"Error in resonator {f0}: {e}")
        return np.array([])

def progress_monitor(progress_dict, total_resonators, resonator_weights, stop_event):
    """Monitor progress across all parallel resonator processes"""
    start_time = time.time()
    total_work = sum(resonator_weights.values())
    last_percent = -1
    
    while not stop_event.is_set():
        time.sleep(0.5)
        
        # Calculate weighted progress
        completed_work = 0
        for resonator_id in range(total_resonators):
            if resonator_id in progress_dict:
                current, total_for_resonator = progress_dict[resonator_id]
                resonator_weight = resonator_weights[resonator_id]
                
                if total_for_resonator > 0:
                    resonator_progress = min(current / total_for_resonator, 1.0)
                    completed_work += resonator_progress * resonator_weight
        
        if total_work > 0:
            percent = int((completed_work / total_work) * 100)
        else:
            percent = 0
        
        if percent != last_percent:
            elapsed = time.time() - start_time
            
            if percent > 0:
                eta_seconds = (elapsed / percent) * (100 - percent)
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "calc..."
            
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
            progress_line = f"[{bar}] {percent:3d}% | {elapsed_str} | ETA: {eta_str}"
            
            print(f"\r{progress_line:<80}", end='', flush=True)
            last_percent = percent
        
        if percent >= 100:
            break
    
    elapsed = time.time() - start_time
    final_line = f"[{'█' * 30}] 100% | {int(elapsed//60)}:{int(elapsed%60):02d} | Complete!"
    print(f"\r{final_line:<80}")
    print()

def process_with_resonator_grid_parallel(signal, fs, clk_resonators, duration, num_processes=None):
    """Process signal with resonator grid using parallel processing"""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Using {num_processes} processes for parallel computation")
    
    # Prepare all resonator tasks
    tasks = []
    resonator_id = 0
    resonator_weights = {}
    
    for clk_freq, freqs in clk_resonators.items():
        print(f"Preparing resonators for clock frequency {clk_freq}")
        sliced_data_resampled = resample_signal(clk_freq, fs, signal)
        actual_samples = len(sliced_data_resampled)
        
        for f0 in freqs:
            tasks.append((f0, clk_freq, sliced_data_resampled, resonator_id))
            resonator_weights[resonator_id] = actual_samples
            resonator_id += 1
    
    print(f"Processing {len(tasks)} resonators in parallel")
    
    # Create shared progress dictionary
    with multiprocessing.Manager() as manager:
        progress_dict = manager.dict()
        stop_event = threading.Event()
        
        # Start progress monitor
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(progress_dict, len(tasks), resonator_weights, stop_event)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            results = Parallel(n_jobs=num_processes, verbose=0)(
                delayed(process_single_resonator)(f0, clk_freq, resampled_signal, progress_dict, res_id)
                for f0, clk_freq, resampled_signal, res_id in tasks
            )
        finally:
            stop_event.set()
            monitor_thread.join(timeout=1)
    
    # Reorganize results
    output = {}
    result_idx = 0
    
    for clk_freq, freqs in clk_resonators.items():
        output[clk_freq] = []
        for f0 in freqs:
            output[clk_freq].append(results[result_idx])
            result_idx += 1
    
    return output

def spikes_event_spectrogram(clk_freq, events, window_ms=10, duration_s=None):
    """Convert spike events to binned counts"""
    window = clk_freq / 1000 * window_ms
    
    if duration_s is None:
        if len(events) == 0:
            return np.array([0])
        duration_s = events[-1] / clk_freq + 1
    
    duration_samples = int(duration_s * clk_freq)
    N = int(np.ceil(duration_samples / window))
    
    bins = np.zeros(N, dtype=int)
    
    if len(events) > 0:
        bin_indices = (events // window).astype(int)
        valid_indices = bin_indices[bin_indices < N]
        
        for idx in valid_indices:
            bins[idx] += 1
    
    return bins

def events_to_max_spectrogram(resonators_by_clk, duration, clk_resonators, signal_file, main_clk=153600):
    """Convert spike events to max spectrogram"""
    is_human_data = 'human' in str(signal_file).lower()
    is_nothing_file = 'nothing' in str(signal_file).lower()
    
    # Get all frequencies
    all_freqs = []
    for clk_freq, freqs in clk_resonators.items():
        all_freqs.extend(freqs)
    
    # Create spectrogram
    max_spikes_spectrogram = np.zeros((len(all_freqs), int(duration * 100)))
    i = 0
    
    for clk_freq, spikes_arrays in resonators_by_clk.items():
        for events in spikes_arrays:
            spikes_spectrogram = spikes_event_spectrogram(clk_freq, events, 10, duration)
            
            if len(spikes_spectrogram) > 0:
                if len(spikes_spectrogram) >= max_spikes_spectrogram.shape[1]:
                    max_spikes_spectrogram[i, :] = spikes_spectrogram[:max_spikes_spectrogram.shape[1]]
                else:
                    max_spikes_spectrogram[i, :len(spikes_spectrogram)] = spikes_spectrogram
                
                # Apply normalization
                max_spikes_spectrogram[i] *= main_clk / clk_freq
                
                # Remove DC and threshold
                max_spikes_spectrogram[i] -= np.mean(max_spikes_spectrogram[i])
                max_spikes_spectrogram[i][max_spikes_spectrogram[i] < 0] = 0
                
                # Apply enhancement
                if np.max(max_spikes_spectrogram[i]) > 0:
                    normalized = max_spikes_spectrogram[i] / np.max(max_spikes_spectrogram[i])
                    
                    if is_nothing_file:
                        if is_human_data:
                            enhanced = np.power(normalized, 0.7)
                        else:
                            enhanced = np.power(normalized, 0.6)
                    else:
                        if is_human_data:
                            enhanced = np.power(normalized, 0.7)
                        else:
                            enhanced = np.power(normalized, 0.5)
                    
                    max_spikes_spectrogram[i] = enhanced * np.max(max_spikes_spectrogram[i])
            
            i += 1
    
    return max_spikes_spectrogram, all_freqs

def spikes_to_bands(spectrogram, frequencies):
    """Group spike spectrogram into frequency bands"""
    corrected_frequencies = np.array(frequencies)
    bands_spectrogram = np.zeros((len(bands), spectrogram.shape[1]))
    
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        band_indices = np.where((corrected_frequencies >= fmin) & (corrected_frequencies < fmax))[0]
        
        if len(band_indices) > 0:
            bands_spectrogram[i] = np.max(spectrogram[band_indices], axis=0)
    
    return bands_spectrogram

def load_and_split_human_file(file_path, human_end_time=32.5, sampling_freq=1000):
    """Load and split the 60-second geophone file"""
    print(f"📂 Loading and splitting file: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        
        if 'amplitude' in data.columns:
            full_signal = data['amplitude'].values
        else:
            full_signal = data.iloc[:, 1].values
        
        print(f"   📊 Loaded {len(full_signal)} samples ({len(full_signal)/sampling_freq:.1f} seconds)")
        
        # Split the signal
        split_sample = int(human_end_time * sampling_freq)
        human_signal = full_signal[:split_sample]
        background_signal = full_signal[split_sample:]
        
        # Normalize
        human_signal = normalize_signal(human_signal)
        background_signal = normalize_signal(background_signal)
        
        print(f"   👤 Human section: {len(human_signal)} samples ({len(human_signal)/sampling_freq:.1f}s)")
        print(f"   🔇 Background section: {len(background_signal)} samples ({len(background_signal)/sampling_freq:.1f}s)")
        
        return human_signal, background_signal
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None

def create_visualization(signal, spikes_bands_spectrogram, duration, section_name, chunk_idx, output_path):
    """Create visualization with raw signal and spikegram"""
    try:
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Raw Signal
        time_axis = np.arange(len(signal)) / SAMPLING_FREQ
        axs[0].plot(time_axis, signal)
        axs[0].set_title(f'{section_name} - Chunk {chunk_idx} - Raw Signal ({duration:.1f}s)', fontsize=14)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: Spikegram
        band_labels = [f'{fmin}-{fmax} ({band})' for band, (fmin, fmax) in bands.items()]
        
        if 'human' in section_name and 'nothing' not in section_name:
            vmax = np.percentile(spikes_bands_spectrogram, 97)
        else:
            vmax = np.percentile(spikes_bands_spectrogram, 99)
        
        im = axs[1].imshow(spikes_bands_spectrogram, aspect='auto', cmap='jet', origin='lower',
                          extent=[0, duration, 0, len(bands)], vmin=0, vmax=vmax)
        axs[1].set_yticks(np.arange(len(band_labels)) + 0.5)
        axs[1].set_yticklabels(band_labels)
        axs[1].set_title(f'{section_name} - Chunk {chunk_idx} - Resonator Spikegram', fontsize=14)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Frequency Band')
        fig.colorbar(im, ax=axs[1], label='Spike Activity', pad=0.01)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def process_signal_section(signal, section_name, output_dir, chunk_duration=30, num_processes=15):
    """Process a signal section using resonator pipeline"""
    print(f"\n🔄 Processing {section_name} section...")
    print("=" * 60)
    
    section_output_dir = os.path.join(output_dir, section_name)
    os.makedirs(section_output_dir, exist_ok=True)
    
    total_duration = len(signal) / SAMPLING_FREQ
    print(f"📊 Section duration: {total_duration:.1f} seconds")
    
    # Determine chunks
    chunk_boundaries = []
    current_pos = 0
    
    while current_pos < total_duration:
        next_pos = min(current_pos + chunk_duration, total_duration)
        chunk_boundaries.append((current_pos, next_pos))
        current_pos = next_pos
        if current_pos >= total_duration:
            break
    
    print(f"📂 Will create {len(chunk_boundaries)} chunks")
    
    # Select resonator grid
    if 'human' in section_name and 'nothing' not in section_name:
        clk_resonators = clk_resonators_human
        print("🧠 Using human-optimized resonator grid")
    else:
        clk_resonators = clk_resonators_background
        print("🔇 Using background-optimized resonator grid")
    
    chunk_results = []
    
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        current_chunk_duration = chunk_end - chunk_start
        
        if current_chunk_duration <= 0:
            break
        
        print(f"\n--- Processing chunk {chunk_idx}: {chunk_start:.1f}s-{chunk_end:.1f}s ---")
        
        # Extract chunk
        start_sample = int(chunk_start * SAMPLING_FREQ)
        end_sample = int(chunk_end * SAMPLING_FREQ)
        chunk_signal = signal[start_sample:end_sample]
        
        if len(chunk_signal) == 0:
            continue
        
        try:
            # Process with resonators
            print(f"⚡ Processing chunk {chunk_idx} with resonator grid...")
            resonator_output = process_with_resonator_grid_parallel(
                chunk_signal, SAMPLING_FREQ, clk_resonators, current_chunk_duration, num_processes
            )
            
            # Create spectrograms
            print(f"📊 Creating spike spectrograms for chunk {chunk_idx}...")
            max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
                resonator_output, current_chunk_duration, clk_resonators, section_name
            )
            
            spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)
            
            # Save results
            chunk_result = {
                'chunk_idx': chunk_idx,
                'start_time': chunk_start,
                'duration': current_chunk_duration,
                'signal': chunk_signal,
                'resonator_outputs': resonator_output,
                'max_spikes_spectrogram': max_spikes_spectrogram,
                'spikes_bands_spectrogram': spikes_bands_spectrogram,
                'all_freqs': all_freqs,
                'section_name': section_name
            }
            
            # Create chunk directory
            chunk_output_dir = os.path.join(section_output_dir, f"chunk_{chunk_idx}")
            os.makedirs(chunk_output_dir, exist_ok=True)
            
            # Save data
            chunk_file = os.path.join(chunk_output_dir, f"chunk_{chunk_idx}_data.pkl")
            with open(chunk_file, 'wb') as f:
                pickle.dump(chunk_result, f)
            
            # Create visualization
            viz_path = os.path.join(chunk_output_dir, f"{section_name}_chunk_{chunk_idx}_visualization.png")
            create_visualization(chunk_signal, spikes_bands_spectrogram, current_chunk_duration, 
                               section_name, chunk_idx, viz_path)
            
            chunk_results.append(chunk_result)
            print(f"✅ Chunk {chunk_idx} processed and saved")
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_idx}: {e}")
            continue
    
    print(f"\n✅ {section_name} processing complete: {len(chunk_results)} chunks saved")
    return chunk_results

def process_human_stepping_file(file_path, human_end_time=32.5, chunk_duration=30, num_processes=15):
    """Main function to process the human stepping file"""
    print("🧠 HUMAN STEPPING FILE PROCESSOR")
    print("=" * 80)
    print(f"📂 Input file: {file_path}")
    print(f"👤 Human stepping: 0 - {human_end_time}s")
    print(f"🔇 Background: {human_end_time}s - {TOTAL_DURATION}s")
    print(f"📁 Output directory: {CHUNKED_OUTPUT_DIR}")
    print("=" * 80)
    
    start_time = time.time()
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    # Load and split
    human_signal, background_signal = load_and_split_human_file(file_path, human_end_time, SAMPLING_FREQ)
    
    if human_signal is None:
        print("❌ Error: Failed to load and split the file")
        return None
    
    os.makedirs(CHUNKED_OUTPUT_DIR, exist_ok=True)
    
    # Process both sections
    print(f"\n🔄 PROCESSING HUMAN STEPPING SECTION")
    human_results = process_signal_section(
        human_signal, "human_new", CHUNKED_OUTPUT_DIR, chunk_duration, num_processes
    )
    
    print(f"\n🔄 PROCESSING BACKGROUND SECTION")
    background_results = process_signal_section(
        background_signal, "human_nothing_new", CHUNKED_OUTPUT_DIR, chunk_duration, num_processes
    )
    
    total_time = time.time() - start_time
    
    print(f"\n✅ PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"👤 Human chunks processed: {len(human_results)}")
    print(f"🔇 Background chunks processed: {len(background_results)}")
    print(f"📁 Human output: {CHUNKED_OUTPUT_DIR}/human_new/")
    print(f"📁 Background output: {CHUNKED_OUTPUT_DIR}/human_nothing_new/")
    
    # Save summary
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
        }
    }
    
    summary_file = os.path.join(CHUNKED_OUTPUT_DIR, "processing_summary.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"📋 Processing summary saved: {summary_file}")
    return summary

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_human_file_simple.py <path_to_csv_file>")
        print("Example: python process_human_file_simple.py /path/to/geophone_60sec.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    print("🧠 HUMAN STEPPING RESONATOR PROCESSOR")
    print("=" * 80)
    print("Processing 60-second geophone file:")
    print("  • 32.5 seconds of human stepping (0-32.5s)")
    print("  • 27.5 seconds of background/no stepping (32.5-60s)")
    print("=" * 80)
    
    # Process the file
    summary = process_human_stepping_file(
        file_path=file_path,
        human_end_time=32.5,
        chunk_duration=30,
        num_processes=15
    )
    
    if summary:
        print("\n🎉 SUCCESS! File processed successfully.")
        print(f"Check the output directories:")
        print(f"  👤 Human: {summary['output_directories']['human']}")
        print(f"  🔇 Background: {summary['output_directories']['background']}") 