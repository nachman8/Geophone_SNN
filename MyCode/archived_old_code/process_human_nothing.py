#!/usr/bin/env python3
"""
SCTN (Spike Continuous Time Neuron) Signal Processor for Human_Nothing Geophone Data
=====================================================================

A specialized version of the SCTN resonator-based signal processing system
that focuses exclusively on processing human_nothing.csv data.
Uses the identical resonator processing pipeline as Geophone_Signal_Processor.py.

Key Features:
- SCTN resonator-based signal processing
- Parallel processing with real-time progress monitoring
- Signal-specific resonator optimization
- Comprehensive visualization and data storage

Author: Seismic Processing Team
Version: 2.0
Framework: SCTN (Spike Continuous Time Neuron) Resonator Processing
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
import warnings
import traceback
import threading
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import resample, spectrogram
from pathlib import Path

# Configuration constants
DATA_DIR = Path.home() / "data"
HUMAN_NOTHING_FILE = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data/human_nothing.csv"
CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output/human_nothing_processed"

print("🧠 SCTN RESONATOR SIGNAL PROCESSING SYSTEM - HUMAN_NOTHING FOCUS")
print("=" * 70)
print(f"🔄 MODE: Processing human_nothing.csv")
print(f"📊 End-to-end signal processing with parallel resonator computation")
print(f"🎯 Complete pipeline from raw signals to resonator spectrograms")
print("=" * 70)

# ========================================================================
# SCTN LIBRARY INTEGRATION
# ========================================================================

# Integration with SCTN (Spike Continuous Time Neuron) framework
sctn_library_path = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_library_path)

# Core SCTN components for resonator processing
from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator
from sctnN.spiking_neuron import create_SCTN, BINARY

warnings.filterwarnings('ignore')

# ========================================================================
# RESONATOR CONFIGURATION AND SIGNAL PROCESSING
# ========================================================================

# Define the resonator grid optimized for human data detection
clk_resonators_human = {
    153600: [
        22.1,
        30.5, 33.9, 34.7, 41.2,
        50.9, 52.6,
        76.3, 63.6,
        95.4
    ]
}

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

def process_signal(signal):
    """Clean signal without normalization - preserve original amplitude scale"""
    
    # Convert to float64 for consistent processing
    signal = np.asarray(signal, dtype=np.float64)
    
    # Handle NaN and infinite values
    nan_count = np.sum(np.isnan(signal))
    inf_count = np.sum(np.isinf(signal))
    
    if nan_count > 0 or inf_count > 0:
        print(f"   ⚠️  Found {nan_count} NaN and {inf_count} infinite values, replacing with zeros")
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure signal has some dynamic range (warn if signal is constant)
    signal_range = np.max(signal) - np.min(signal)
    if signal_range == 0:
        print(f"   ⚠️  Warning: Signal has no dynamic range (constant value: {np.mean(signal):.3f})")
    else:
        # Check for extremely large amplitudes that might cause issues
        max_abs = np.max(np.abs(signal))
        if max_abs > 1e6:
            print(f"   ⚠️  Warning: Very large amplitudes detected (max: {max_abs:.2e})")
            print(f"      SCTN resonators will process with original scale")
        elif max_abs > 1e3:
            print(f"   ℹ️  Large amplitudes detected (max: {max_abs:.2e}) - processing with original scale")
        
        print(f"   📊 Signal range: {np.min(signal):.3f} to {np.max(signal):.3f} (amplitude preserved)")
    
    # Return signal in original scale without normalization
    return signal

def display_signal_stats(signal, signal_name="Signal"):
    """Display comprehensive statistics for the signal"""
    if len(signal) == 0:
        print(f"   📊 {signal_name}: Empty signal")
        return
    
    stats = {
        'length': len(signal),
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.max(signal) - np.min(signal),
        'rms': np.sqrt(np.mean(signal**2))
    }
    
    print(f"   📊 {signal_name} Statistics:")
    print(f"      Length: {stats['length']:,} samples")
    print(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}] (span: {stats['range']:.4f})")
    print(f"      Mean: {stats['mean']:.4f}, RMS: {stats['rms']:.4f}, Std: {stats['std']:.4f}")

def resample_signal(f_new, f_source, data):
    """
    Resample signal to match a new frequency
    """
    n_samples_orig = data.shape[0]
    n_samples_new = int(n_samples_orig * f_new / f_source)

    # Resample the signal
    return resample(data, n_samples_new)

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

# ========================================================================
# SCTN RESONATOR PROCESSING ENGINE
# ========================================================================

def process_single_resonator(f0, clk_freq, resampled_signal, progress_dict=None, resonator_id=None):
    """
    Process a single SCTN resonator with progress tracking and memory management
    """
    try:
        # Force garbage collection before processing
        gc.collect()
        
        # Get closest resonator function
        resonator_func, actual_freq = get_closest_resonator(f0)

        # Create SCTN resonator
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
        
        # Clean up resonator to free memory
        del my_resonator
        gc.collect()

        return output_spikes

    except Exception as e:
        # Clean up on error
        gc.collect()
        return np.array([])

def process_with_resonator_grid_parallel(signal, fs, clk_resonators, duration, num_processes=None):
    """
    Process signal with SCTN resonator grid using parallel processing
    """
    if num_processes is None:
        num_processes = 10  # Default to 10 processors
    else:
        num_processes = min(num_processes, 15)  # Cap at 15 processors max

    print(f"Using {num_processes} processes for parallel SCTN resonator computation")

    # Prepare all resonator tasks for parallel processing
    tasks = []
    resonator_id = 0
    resonator_weights = {}  # Store actual sample count for each resonator
    
    for clk_freq, freqs in clk_resonators.items():
        print(f"Preparing SCTN resonators for clock frequency {clk_freq}")
        # Resample signal to match clock frequency (without windowing)
        sliced_data_resampled = resample_signal(clk_freq, fs, signal)
        actual_samples = len(sliced_data_resampled)
        
        # Add tasks for all resonators at this clock frequency
        for f0 in freqs:
            tasks.append((f0, clk_freq, sliced_data_resampled, resonator_id))
            resonator_weights[resonator_id] = actual_samples  # Store actual work for this resonator
            resonator_id += 1

    print(f"Processing {len(tasks)} SCTN resonators in parallel across all clock frequencies")
    
    # Create shared progress dictionary using Manager
    with multiprocessing.Manager() as manager:
        progress_dict = manager.dict()
        stop_event = threading.Event()
        
        # Start progress monitor in a separate thread with weighted progress
        monitor_thread = threading.Thread(
            target=progress_monitor, 
            args=(progress_dict, len(tasks), resonator_weights, stop_event)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            from joblib import Parallel, delayed
            
            results = Parallel(n_jobs=num_processes, verbose=0, backend='loky')(
                delayed(process_single_resonator)(f0, clk_freq, resampled_signal, progress_dict, res_id) 
                for f0, clk_freq, resampled_signal, res_id in tasks
            )
                
        except Exception as e:
            print(f"⚠️  Error in parallel processing: {e}")
            # Emergency fallback: sequential processing
            print("🔄 Falling back to sequential processing...")
            results = []
            for f0, clk_freq, resampled_signal, res_id in tasks:
                try:
                    result = process_single_resonator(f0, clk_freq, resampled_signal, None, res_id)
                    results.append(result)
                except Exception as seq_e:
                    print(f"   ⚠️  Error in resonator {f0}Hz: {seq_e}")
                    results.append(np.array([]))
        finally:
            # Stop the progress monitor
            stop_event.set()
            monitor_thread.join(timeout=1)
            print("✅ Parallel processing completed successfully")

    # Reorganize results back into the original structure
    output = {}
    result_idx = 0
    
    for clk_freq, freqs in clk_resonators.items():
        output[clk_freq] = []
        for f0 in freqs:
            output[clk_freq].append(results[result_idx])
            result_idx += 1

    return output

def spikes_event_spectrogram(clk_freq, events, window_ms=10, duration_s=None):
    """
    Convert spike events to binned counts
    """
    window = clk_freq / 1000 * window_ms

    if duration_s is None:
        if len(events) == 0:
            return np.array([0])
        duration_s = events[-1] / clk_freq + 1

    duration_samples = int(duration_s * clk_freq)
    N = int(np.ceil(duration_samples / window))

    bins = np.zeros(N, dtype=int)

    if len(events) > 0:
        # Find which bin each event belongs to
        bin_indices = (events // window).astype(int)

        # Only use valid indices
        valid_indices = bin_indices[bin_indices < N]

        # Count events in each bin
        for idx in valid_indices:
            bins[idx] += 1

    return bins

def events_to_max_spectrogram(resonators_by_clk, duration, clk_resonators, signal_file, main_clk=153600):
    """
    Convert spike events to max spectrogram with DC removal, thresholding, and amplitude enhancement for robust detection
    """
    # Detect data type and signal strength for adaptive parameters
    is_human_data = 'human' in str(signal_file).lower()
    is_nothing_file = 'nothing' in str(signal_file).lower()
    
    # Get all frequencies from resonator grid
    all_freqs = []
    for clk_freq, freqs in clk_resonators.items():
        all_freqs.extend(freqs)

    # Create empty spectrogram - use 10ms window for fine detail
    max_spikes_spectrogram = np.zeros((len(all_freqs), int(duration * 100)))
    i = 0

    for clk_freq, spikes_arrays in resonators_by_clk.items():
        for events in spikes_arrays:
            # Convert events to binned spike counts
            spikes_spectrogram = spikes_event_spectrogram(clk_freq, events, 10, duration)

            # Ensure we have enough bins
            if len(spikes_spectrogram) > 0:
                # Match dimensions
                if len(spikes_spectrogram) >= max_spikes_spectrogram.shape[1]:
                    max_spikes_spectrogram[i, :] = spikes_spectrogram[:max_spikes_spectrogram.shape[1]]
                else:
                    max_spikes_spectrogram[i, :len(spikes_spectrogram)] = spikes_spectrogram

                # Apply clock frequency normalization
                max_spikes_spectrogram[i] *= main_clk / clk_freq
                
                # Remove DC component for better contrast (always use mean - standard approach)
                max_spikes_spectrogram[i] -= np.mean(max_spikes_spectrogram[i])
                
                # THRESHOLD: Set negative values to zero
                max_spikes_spectrogram[i][max_spikes_spectrogram[i] < 0] = 0
                
                # AMPLITUDE ENHANCEMENT: Apply adaptive power function
                if np.max(max_spikes_spectrogram[i]) > 0:
                    # Normalize to [0,1] then apply power function for enhancement
                    normalized = max_spikes_spectrogram[i] / np.max(max_spikes_spectrogram[i])
                    
                    # Check for both nothing files (background signals)
                    is_nothing_here = 'nothing' in str(signal_file).lower()
                    
                    if is_nothing_here:
                        # For both nothing files: gentler enhancement for background signals
                        if is_human_data:
                            enhanced = np.power(normalized, 0.7)  # Moderate for human_nothing
                        else:
                            enhanced = np.power(normalized, 0.6)  # Moderate for car_nothing
                    else:
                        # For active signal files: adjusted enhancement
                        if is_human_data:
                            enhanced = np.power(normalized, 0.7)  # More aggressive for human (was 0.6)
                        else:
                            enhanced = np.power(normalized, 0.5)  # Original for car data
                    
                    max_spikes_spectrogram[i] = enhanced * np.max(max_spikes_spectrogram[i])

            i += 1

    return max_spikes_spectrogram, all_freqs

def spikes_to_bands(spectrogram, frequencies):
    """
    Group spike spectrogram into frequency bands using max for stronger signal visibility.
    """
    # Use original frequencies without correction
    corrected_frequencies = np.array(frequencies)

    # Create band spectrogram
    bands_spectrogram = np.zeros((len(bands), spectrogram.shape[1]))

    # Fill with data for each band
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        # Find indices of frequencies in this band
        band_indices = np.where((corrected_frequencies >= fmin) & (corrected_frequencies < fmax))[0]

        if len(band_indices) > 0:
            # Use max for stronger signal visibility (now safe with thresholding)
            bands_spectrogram[i] = np.max(spectrogram[band_indices], axis=0)

    return bands_spectrogram

def progress_monitor(progress_dict, total_resonators, resonator_weights, stop_event):
    """
    Monitor progress across all parallel SCTN resonator processes with smooth progress bar
    """
    start_time = time.time()
    
    # Calculate total work (sum of all samples across all resonators)
    total_work = sum(resonator_weights.values())
    
    print(f"\nProcessing {total_resonators} SCTN resonators in parallel:")
    for resonator_id, samples in resonator_weights.items():
        clk_freq = 153600  # Only 153600 supported now
        print(f"  Resonator {resonator_id}: {samples:,} samples @ {clk_freq} Hz")
    print(f"Total work: {total_work:,} samples")
    print(f"Starting parallel processing...")
    
    last_percent = -1
    progress_started = False
    
    while not stop_event.is_set():
        time.sleep(0.5)  # Check every 0.5 seconds
        
        # Calculate weighted progress
        completed_work = 0
        
        for resonator_id in range(total_resonators):
            if resonator_id in progress_dict:
                current, total_for_resonator = progress_dict[resonator_id]
                resonator_weight = resonator_weights[resonator_id]
                
                # Calculate work completed by this resonator
                if total_for_resonator > 0:
                    resonator_progress = min(current / total_for_resonator, 1.0)
                    completed_work += resonator_progress * resonator_weight
        
        # Calculate overall percentage based on weighted work
        if total_work > 0:
            percent = int((completed_work / total_work) * 100)
        else:
            percent = 0
        
        # Only print when percentage changes AND above 0%
        if percent != last_percent and percent > 0:
            elapsed = time.time() - start_time
            
            if percent > 0:
                eta_seconds = (elapsed / percent) * (100 - percent)
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "calc..."
            
            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Format progress line
            elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
            if percent >= 100:
                progress_line = f"[{bar}] 100% | {elapsed_str} | Complete!"
            else:
                progress_line = f"[{bar}] {percent:3d}% | {elapsed_str} | ETA: {eta_str}"
            
            # FIXED: Always use \r to overwrite the same line
            print(f"\r{progress_line:<80}", end='', flush=True)
            progress_started = True
            last_percent = percent
        
        # Check if we're done
        if percent >= 100:
            break
    
    # Ensure we end with a newline when complete
    if progress_started:
        print()  # New line after completion

# ========================================================================
# FILE PROCESSING AND CHUNKED PIPELINE
# ========================================================================

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
        
        # process the signal
        signal = process_signal(signal)
        
        # Create time axis
        time = np.arange(len(signal)) / sampling_freq
        
        actual_duration = len(signal) / sampling_freq
        
        print(f"Loaded chunk {start_time:.1f}-{start_time + actual_duration:.1f}s: {len(signal)} samples")
        
        return signal, time, actual_duration
        
    except Exception as e:
        print(f"Error loading chunk from {file_path}: {e}")
        return None, None, None

def process_chunk(signal, time, actual_duration, chunk_idx, output_dir, num_processes=15):
    """
    Process a single chunk of data with SCTN resonator grid
    """
    print(f"\n--- Processing chunk {chunk_idx} ---")
    
    if signal is None:
        print(f"Failed to process chunk {chunk_idx}: No signal data")
        return None
    
    # Use human resonator grid
    clk_resonators = clk_resonators_human
    
    # Process with resonator grid
    print(f"Processing chunk {chunk_idx} with SCTN resonator grid...")
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
            "human_nothing.csv"
        )
        
        # Group by frequency bands
        spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)
        
        # Save chunk results
        chunk_results = {
            'chunk_idx': chunk_idx,
            'duration': actual_duration,
            'signal': signal,
            'time': time,
            'resonator_outputs': output,
            'max_spikes_spectrogram': max_spikes_spectrogram,
            'spikes_bands_spectrogram': spikes_bands_spectrogram,
            'all_freqs': all_freqs,
            'file_path': HUMAN_NOTHING_FILE
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
        traceback.print_exc()
        return None

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
        
        # Plot 3: Spikegram (SCTN Resonator Output) 
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
        
        # Adaptive visualization parameters - for human_nothing
        vmax = np.percentile(spikes_bands_spectrogram, 97)
        
        im2 = axs[2].imshow(spikes_bands_spectrogram, aspect='auto', cmap='jet', origin='lower',
                          extent=[0, duration, 0, len(bands)], vmin=0, vmax=vmax)
        axs[2].set_yticks(np.arange(len(band_labels)) + 0.5)
        axs[2].set_yticklabels(band_labels)
        axs[2].set_title(f'Chunk {chunk_idx} - SCTN Resonator-based Spikegram', fontsize=14)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Frequency Band')
        fig.colorbar(im2, ax=axs[2], label='Spike Activity', pad=0.01)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"chunk_{chunk_idx}_visualization.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Chunk {chunk_idx} visualization (with FFT + Spikegram) saved to {plot_file}")
        
        # Clean up visualization data
        del fig, axs, fft_bin_spectogram, f, t, Sxx
        gc.collect()
        
    except Exception as e:
        print(f"Error creating chunk visualization: {e}")
        traceback.print_exc()

def process_human_nothing_file(chunk_duration=30, num_processes=10):
    """
    Process the human_nothing.csv file with SCTN resonator grid
    """
    print("\n🧠 HUMAN_NOTHING FILE PROCESSING")
    print("=" * 60)
    
    # Get total file duration
    file_path = HUMAN_NOTHING_FILE
    total_duration = get_file_duration(file_path)
    
    if total_duration is None:
        print(f"Failed to get duration for {file_path}")
        return None
    
    # Calculate chunk boundaries
    chunk_boundaries = []
    current_pos = 0
    
    while current_pos < total_duration:
        next_pos = current_pos + chunk_duration
        chunk_boundaries.append((current_pos, min(next_pos, total_duration)))
        current_pos = next_pos
        
        if current_pos >= total_duration:
            break
    
    num_chunks = len(chunk_boundaries)
    print(f"File duration: {total_duration:.1f}s, will process in {num_chunks} chunks of {chunk_duration}s each")
    
    # Create output directory
    os.makedirs(CHUNKED_OUTPUT_DIR, exist_ok=True)
    
    chunk_results = []
    
    # Process each chunk
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        current_chunk_duration = chunk_end - chunk_start
        
        if current_chunk_duration <= 0:
            break
        
        # Load chunk data
        signal, time, actual_duration = load_chunk_data(
            file_path, chunk_start, current_chunk_duration, sampling_freq=1000
        )
        
        # Process chunk
        result = process_chunk(
            signal, time, actual_duration, chunk_idx, 
            CHUNKED_OUTPUT_DIR, num_processes
        )
        
        if result is not None:
            chunk_results.append(result)
        
        # Clean up after each chunk
        gc.collect()
    
    print(f"\n✅ File {file_path} processed in {len(chunk_results)} chunks")
    
    # Save chunk index
    index_file = os.path.join(CHUNKED_OUTPUT_DIR, "chunk_index.pkl")
    chunk_index = {
        'file_path': str(file_path),
        'total_duration': total_duration,
        'chunk_duration': chunk_duration,
        'num_chunks': len(chunk_results),
        'chunk_boundaries': chunk_boundaries,
        'chunk_files': [os.path.join(CHUNKED_OUTPUT_DIR, f"chunk_{i}", f"chunk_{i}_data.pkl") 
                       for i in range(len(chunk_results))]
    }
    
    with open(index_file, 'wb') as f:
        pickle.dump(chunk_index, f)
    
    print(f"Chunk index saved to {index_file}")
    
    return chunk_index

# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    print("\n🚀 Starting human_nothing.csv processing pipeline...")
    process_human_nothing_file(chunk_duration=30, num_processes=10)
    print("✅ Processing complete!")
