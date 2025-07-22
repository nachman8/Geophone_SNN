#!/usr/bin/env python3
"""
Resonator-Based Processing for Geophone Signals
==========================================================================

A system for processing geophone signals using resonator-based analysis.
Processes signals through a grid of resonators and generates spectrograms
for visualization and further analysis.

Key Features:
- Signal preprocessing and normalization
- Parallel resonator grid processing
- Frequency band analysis
- Chunked file processing for memory efficiency
- Visualization of spectrograms and resonator outputs

Author: Seismic Processing Team
Version: 1.0
Framework: Resonator-Based Signal Processing
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample, spectrogram
from joblib import Parallel, delayed
import multiprocessing
import os
import time
import threading
import pickle
from collections import Counter
from pathlib import Path
import gc
import traceback

import sys
# Integration with SCTN (Spiking Cellular Temporal Neural) framework for ensemble architectures
sctn_library_path = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_library_path)
# Import resonator functions from sctnN
from sctnN.resonator import *
from sctnN.spiking_neuron import *

# Configuration constants
PROJECT_PATH = Path("/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project")
DATA_DIR = PROJECT_PATH / "data"
CHUNKED_OUTPUT_DIR = PROJECT_PATH / "MyCode/chunked_output_not_norm"

# Processing configuration
LOAD_FROM_CHUNKED = False  # True: Load pre-computed features, False: Full pipeline
PROCESS_HUMAN_FILE_OTHER = True  # True: Process 40Sen_30Sec_stomping_30sec_quiet.csv through full pipeline
PROCESS_HUMAN_FILE = False  # True: Process human.csv through full pipeline
PROCESS_CAR_FILE = False  # True: Process car.csv through full pipeline
PROCESS_CAR_NOTHING_FILE = False  # True: Process car_nothing.csv through full pipeline
PROCESS_HUMAN_NOTHING_FILE = False  # True: Process human_nothing.csv through full pipeline

# ========================================================================
# SCTN LIBRARY INTEGRATION
# ========================================================================



# Core ensemble components from SCTN framework
from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator
from sctnN.spiking_neuron import create_SCTN, BINARY

import warnings
warnings.filterwarnings('ignore')
# ========================================================================

# ========================================================================


print("🔄 RESONATOR-BASED SIGNAL PROCESSING SYSTEM")

# Define separate resonator grids for different data types
clk_resonators_car = {
    153600: [
        # LOW_FREQ coverage for car
        22.1, 28.8,
        # Enhanced CAR coverage (30-48 Hz) - all available for better car detection
        30.5, 34.7, 37.2, 40.2, 43.6,  47.7,
        # MID_GAP coverage
        52.6, 58.7,
        # Reduced HUMAN coverage - keep some for comparison
        63.6, 69.4, 76.3,
        # HIGH_FREQ coverage
        89.8, 95.4
    ]
}

clk_resonators_human = {
    153600: [
        # Available LOW_FREQ coverage (20-30 Hz) - focus on human activity
        22.1,
        # Reduced CAR coverage (30-48 Hz) - keep minimal but essential
        30.5, 33.9, 34.7, 41.2,
        # Enhanced MID_GAP coverage (48-60 Hz) - all available
        50.9, 52.6,
        # ALL available HUMAN_PEAK and HUMAN_TAIL coverage (60-85 Hz)
        76.3, 63.6,
        # Minimal HIGH_FREQ coverage (85-100 Hz)
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

# Auto-detect data type and select appropriate resonator grid
def get_resonator_grid(signal_file):
    """
    Automatically select the appropriate resonator grid based on the signal file name
    """
    file_name = str(signal_file).lower()
    if 'human' in file_name:
        print("Detected HUMAN data - using human-optimized resonator grid")
        return clk_resonators_human
    elif 'car' in file_name:
        print("Detected CAR data - using car-optimized resonator grid")
        return clk_resonators_car
    else:
        print("Unknown data type - defaulting to car resonator grid")
        return clk_resonators_car

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

def normalize_signal(signal):
    """Normalize signal to [-1, 1] range"""
    signal_min, signal_max = np.min(signal), np.max(signal)
    if signal_max > signal_min:
        return 2 * (signal - signal_min) / (signal_max - signal_min) - 1
    return np.zeros_like(signal)

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

# Function to process a single resonator in parallel
def process_single_resonator(f0, clk_freq, resampled_signal, progress_dict=None, resonator_id=None):
    """
    Process a single resonator with progress tracking
    """
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
        return np.array([])

def process_with_resonator_grid_parallel(signal, fs, clk_resonators, duration, num_processes=None):
    """
    Process signal with resonator grid using parallel processing with real-time progress tracking
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count() - 1

    print(f"Using {num_processes} processes for parallel computation")

    # Prepare all resonator tasks for parallel processing
    tasks = []
    resonator_id = 0
    resonator_weights = {}  # Store actual sample count for each resonator
    
    for clk_freq, freqs in clk_resonators.items():
        print(f"Preparing resonators for clock frequency {clk_freq}")
        # Resample signal to match clock frequency (without windowing)
        sliced_data_resampled = resample_signal(clk_freq, fs, signal)
        actual_samples = len(sliced_data_resampled)
        
        # Add tasks for all resonators at this clock frequency
        for f0 in freqs:
            tasks.append((f0, clk_freq, sliced_data_resampled, resonator_id))
            resonator_weights[resonator_id] = actual_samples  # Store actual work for this resonator
            resonator_id += 1

    print(f"Processing {len(tasks)} resonators in parallel across all clock frequencies")
    
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
            # Use parallel processing to process all resonators
            results = Parallel(n_jobs=num_processes)(
                delayed(process_single_resonator)(
                    f0, clk_freq, resampled_signal, progress_dict, resonator_id
                )
                for f0, clk_freq, resampled_signal, resonator_id in tasks
            )
        except Exception as e:
            stop_event.set()  # Signal the monitor thread to stop
            monitor_thread.join(timeout=1.0)  # Wait for thread to finish but with timeout
            raise e  # Re-raise the exception
        finally:
            stop_event.set()  # Signal the monitor thread to stop
            monitor_thread.join(timeout=1.0)  # Wait for thread to finish but with timeout

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
    Monitor progress across all parallel resonator processes with weighted progress based on actual work
    """
    start_time = time.time()
    
    # Calculate total work (sum of all samples across all resonators)
    total_work = sum(resonator_weights.values())
    
    print(f"\nProcessing {total_resonators} resonators in parallel:")
    for resonator_id, samples in resonator_weights.items():
        clk_freq = 153600  # Only 153600 supported now
        print(f"  Resonator {resonator_id}: {samples:,} samples @ {clk_freq} Hz")
    print(f"Total work: {total_work:,} samples")
    
    last_percent = -1
    
    
    while not stop_event.is_set():
        time.sleep(0.5)  # Check every 0.5 seconds
        
        # Calculate weighted progress
        completed_work = 0
        completed_resonators = 0
        
        try:
            for resonator_id in range(total_resonators):
                try:
                    if resonator_id in progress_dict:
                        current, total_for_resonator = progress_dict[resonator_id]
                        resonator_weight = resonator_weights[resonator_id]
                        
                        # Calculate work completed by this resonator
                        if total_for_resonator > 0:
                            resonator_progress = min(current / total_for_resonator, 1.0)
                            completed_work += resonator_progress * resonator_weight
                            
                            if resonator_progress >= 1.0:
                                completed_resonators += 1
                except Exception:
                    # Ignore errors accessing the dict - connection might be closing
                    pass
        except Exception:
            # If we can't access the dictionary at all, just break the loop
            break
        
        # Calculate overall percentage based on weighted work
        if total_work > 0:
            percent = int((completed_work / total_work) * 100)
        else:
            percent = 0
        
        # Only print when percentage changes
        if percent != last_percent:
            elapsed = time.time() - start_time
            
            if percent > 0:
                eta_seconds = (elapsed / percent) * (100 - percent)
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "calc..."
            
            # Create progress bar (shorter to prevent wrapping)
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Compact format to prevent line wrapping
            elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
            progress_line = f"[{bar}] {percent:3d}% | {elapsed_str} | ETA: {eta_str}"
            
            # Clear line and print progress (add padding to clear previous content)
            print(f"\r{progress_line:<80}", end='', flush=True)
            
            last_percent = percent
        
        # Check if we're done (all work completed)
        if percent >= 100:
            break
    
    try:
        # Final progress update with extra padding to clear the line
        elapsed = time.time() - start_time
        final_line = f"[{'█' * 30}] 100% | {int(elapsed//60)}:{int(elapsed%60):02d} | Complete!"
        print(f"\r{final_line:<80}")
        print()  # New line after completion
    except Exception:
        # In case of any error during cleanup, just silently exit
        pass

def get_file_duration(file_path, sampling_freq=1000):
    """
    Get the total duration of a file without loading the entire file into memory
    """
    try:
        # Read just the first few rows to understand structure
        sample_data = pd.read_csv(file_path, nrows=10)
        
        # Get total row count efficiently
        with open(file_path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # -1 for header
        
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
            signal = chunk_data.iloc[:, 0].values  # Use first column
        
        # Store raw signal for visualization
        raw_signal = signal.copy()
        
        # Normalize the signal for processing
        signal = normalize_signal(signal)
        
        # Create time axis
        time = np.arange(len(signal)) / sampling_freq
        
        actual_duration = len(signal) / sampling_freq
        
        print(f"Loaded chunk {start_time:.1f}-{start_time + actual_duration:.1f}s: {len(signal)} samples")
        
        return signal, time, actual_duration, raw_signal
        
    except Exception as e:
        print(f"Error loading chunk from {file_path}: {e}")
        return None, None, None, None

def process_file_chunk(file_path, chunk_start, chunk_duration, chunk_idx, 
                      output_dir, num_processes=15):
    """
    Process a single chunk of a file and save intermediate results
    """
    print(f"\n--- Processing chunk {chunk_idx}: {chunk_start:.1f}s-{chunk_start + chunk_duration:.1f}s ---")
    
    # Load chunk data
    signal, time, actual_duration, raw_signal = load_chunk_data(file_path, chunk_start, chunk_duration)
    
    if signal is None:
        print(f"Failed to load chunk {chunk_idx}")
        return None
    
    # Auto-detect and select appropriate resonator grid
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
            'raw_signal': raw_signal,  # Store raw signal for visualization
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
        traceback.print_exc()
        return None

def create_chunk_visualization(chunk_results, output_dir):
    """
    Create visualization for a single chunk with both FFT spectrogram and spikegram
    """
    try:
        signal = chunk_results['signal']  # Normalized signal
        raw_signal = chunk_results.get('raw_signal', signal)  # Get raw signal if available, otherwise use normalized
        time = chunk_results['time']
        spikes_bands_spectrogram = chunk_results['spikes_bands_spectrogram']
        duration = chunk_results['duration']
        chunk_idx = chunk_results['chunk_idx']
        
        # Create a comprehensive visualization for the chunk
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1.5, 1.5]})
        
        # Plot 1: Raw Signal (use non-normalized signal here as requested)
        axs[0].plot(time, raw_signal)
        axs[0].set_title(f'Chunk {chunk_idx} - Raw Signal (Non-Normalized) ({duration:.1f}s)', fontsize=14)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: FFT Spectrogram (use normalized signal for processing)
        print(f"Computing FFT spectrogram for chunk {chunk_idx}...")
        f, t, Sxx = compute_fft_spectrogram(signal, 1000, fmin=1, fmax=100, plot=False)
        
        # Create band labels
        band_labels = [f'{fmin}-{fmax} ({band})' for band, (fmin, fmax) in bands.items()]
        
        # Convert FFT spectrogram to band-based representation
        fft_bin_spectogram = np.zeros((len(bands), len(t)))
        for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            band_mask = (f >= fmin) & (f < fmax)
            if np.any(band_mask):
                fft_bin_spectogram[i] = np.mean(Sxx[band_mask], axis=0)
        
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
        # Downsample to match FFT spectrogram time resolution (exact same logic as visualize_comparison)
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
        
        # Adaptive visualization parameters
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
        traceback.print_exc()

def process_file_in_chunks(file_path, chunk_duration=30, num_processes=15, min_chunk_size=10):
    """
    Process a single file in chunks to manage memory usage
    Small leftover chunks are added to the previous chunk to avoid tiny chunks
    """
    print(f"\n🔄 PROCESSING FILE IN CHUNKS: {file_path}")
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunked_output_not_norm", file_stem)
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
        gc.collect()
    
    print(f"\n✅ File {file_path} processed in {len(chunk_results)} chunks")
    
    # Save chunk index for this file (for backward compatibility)
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
    
    return chunk_index

def visualize_resonator_output(file_path, output_dir=None):
    """
    Process a file with resonators and visualize the results
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Processing and visualizing: {file_path}")
    
    # Process the file in chunks
    chunk_index = process_file_in_chunks(file_path)
    
    if chunk_index is None:
        print("❌ Failed to process file")
        return
    
    print(f"\n✅ Visualization complete. Results saved to {output_dir}")
    return chunk_index

# Function to process the multi-sensor file with separate handling for human and background
def process_multi_sensor_file(file_path, human_end_time=32.5, sampling_freq=1000):
    """
    Process the multi-sensor file (40Sen_30Sec_stomping_30sec_quiet.csv) that contains 40 columns
    of sensor data, each representing 60 seconds (32.5s human + 27.5s background).
    
    Args:
        file_path (str): Path to the CSV file with multiple sensor columns
        human_end_time (float): Time in seconds where human stepping ends
        sampling_freq (int): Sampling frequency in Hz
        
    Returns:
        dict: Summary of processing results
    """
    print("\n" + "=" * 80)
    print("🔄 MULTI-SENSOR FILE PROCESSOR")
    print("=" * 80)
    print(f"📂 Processing multi-sensor file: {file_path}")
    print(f"👤 Human stepping: 0 - {human_end_time}s")
    print(f"🔇 Background: {human_end_time}s - 60.0s")
    
    try:
        # Load the full file
        print(f"📊 Loading multi-sensor data...")
        data = pd.read_csv(file_path)
        
        # Check column count to confirm it's the multi-sensor format
        column_count = len(data.columns)
        print(f"📈 Found {column_count} columns of sensor data")
        
        # Calculate split point based on human_end_time
        split_sample = int(human_end_time * sampling_freq)
        
        # Create directories for human and background sections
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunked_output_not_norm")
        human_output_dir = os.path.join(output_dir, "human_new")
        background_output_dir = os.path.join(output_dir, "human_nothing_new")
        os.makedirs(human_output_dir, exist_ok=True)
        os.makedirs(background_output_dir, exist_ok=True)
        
        print(f"📁 Output directories created:")
        print(f"   👤 Human: {human_output_dir}")
        print(f"   🔇 Background: {background_output_dir}")
        
        # Use the human-optimized resonator grid for both parts
        human_resonator_grid = {
            153600: [
                # Human-optimized resonators
                22.1,  # LOW_FREQ
                30.5, 33.9, 34.7, 41.2,  # CAR coverage
                50.9, 52.6,  # MID_GAP
                76.3, 63.6,  # HUMAN_PEAK and HUMAN_TAIL
                95.4  # HIGH_FREQ
            ]
        }
        
        human_chunks_processed = 0
        background_chunks_processed = 0
        
        # Process each sensor column
        for col_idx, column_name in enumerate(data.columns):
            print(f"\n🔄 Processing sensor {col_idx+1}/{column_count}: {column_name}")
            
            try:
                # Get data for this sensor
                sensor_data = data[column_name].values
                
                # Handle NaN values by replacing them with zeros or interpolation
                nan_count = np.isnan(sensor_data).sum()
                if nan_count > 0:
                    print(f"⚠️  Found {nan_count} NaN values in column {column_name}, replacing with interpolation")
                    # Fill NaN values using forward fill, then backward fill for any remaining NaNs
                    sensor_data_series = pd.Series(sensor_data)
                    sensor_data = sensor_data_series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').values
                    
                    # If still have NaNs (e.g., all NaN column), replace with zeros
                    if np.isnan(sensor_data).any():
                        sensor_data = np.nan_to_num(sensor_data)
                
                # Split into human and background parts
                human_signal = sensor_data[:split_sample]
                background_signal = sensor_data[split_sample:]
                
                # Create time axes
                time_human = np.arange(len(human_signal)) / sampling_freq
                time_background = np.arange(len(background_signal)) / sampling_freq
                
                # Store raw signals for visualization
                raw_human_signal = human_signal.copy()
                raw_background_signal = background_signal.copy()
                
                # Normalize signals
                human_signal = normalize_signal(human_signal)
                background_signal = normalize_signal(background_signal)
                
                # Process human part
                print(f"👤 Processing human stepping part for sensor {col_idx+1}...")
                
                # Create chunk result for human part
                human_chunk = {
                    'chunk_idx': col_idx,
                    'start_time': 0,
                    'duration': human_end_time,
                    'signal': human_signal,
                    'raw_signal': raw_human_signal,
                    'time': time_human,
                    'file_path': str(file_path),
                    'section_name': 'human_new',
                    'sensor_column': column_name
                }
                
                # Process with resonator grid
                try:
                    # Process with resonator grid using the human-optimized resonators
                    human_output = process_with_resonator_grid_parallel(
                        human_signal,
                        sampling_freq,
                        human_resonator_grid,
                        human_end_time,
                        num_processes=15
                    )
                    
                    # Create spectrograms
                    human_spikes_spectrogram, human_all_freqs = events_to_max_spectrogram(
                        human_output,
                        human_end_time,
                        human_resonator_grid,
                        "human"  # Signal type identifier
                    )
                    
                    # Group by frequency bands
                    human_bands_spectrogram = spikes_to_bands(human_spikes_spectrogram, human_all_freqs)
                    
                    # Add to chunk result
                    human_chunk['resonator_outputs'] = human_output
                    human_chunk['max_spikes_spectrogram'] = human_spikes_spectrogram
                    human_chunk['spikes_bands_spectrogram'] = human_bands_spectrogram
                    human_chunk['all_freqs'] = human_all_freqs
                    
                    # Save human chunk
                    human_chunk_dir = os.path.join(human_output_dir, f"sensor_{col_idx+1}")
                    os.makedirs(human_chunk_dir, exist_ok=True)
                    
                    human_chunk_file = os.path.join(human_chunk_dir, f"chunk_{col_idx+1}_data.pkl")
                    with open(human_chunk_file, 'wb') as f:
                        pickle.dump(human_chunk, f)
                    
                    # Create visualization
                    create_chunk_visualization(human_chunk, human_chunk_dir)
                    
                    human_chunks_processed += 1
                    print(f"✅ Human part of sensor {col_idx+1} processed and saved")
                    
                except Exception as e:
                    print(f"❌ Error processing human part of sensor {col_idx+1}: {str(e)}")
                    traceback.print_exc()
                
                # Process background part
                print(f"🔇 Processing background part for sensor {col_idx+1}...")
                
                # Create chunk result for background part
                background_chunk = {
                    'chunk_idx': col_idx,
                    'start_time': human_end_time,
                    'duration': 60.0 - human_end_time,
                    'signal': background_signal,
                    'raw_signal': raw_background_signal,
                    'time': time_background,
                    'file_path': str(file_path),
                    'section_name': 'human_nothing_new',
                    'sensor_column': column_name
                }
                
                try:
                    # Process with resonator grid using the human-optimized resonators (for consistency)
                    background_output = process_with_resonator_grid_parallel(
                        background_signal,
                        sampling_freq,
                        human_resonator_grid,
                        60.0 - human_end_time,
                        num_processes=15
                    )
                    
                    # Create spectrograms
                    background_spikes_spectrogram, background_all_freqs = events_to_max_spectrogram(
                        background_output,
                        60.0 - human_end_time,
                        human_resonator_grid,
                        "human_nothing"  # Signal type identifier
                    )
                    
                    # Group by frequency bands
                    background_bands_spectrogram = spikes_to_bands(background_spikes_spectrogram, background_all_freqs)
                    
                    # Add to chunk result
                    background_chunk['resonator_outputs'] = background_output
                    background_chunk['max_spikes_spectrogram'] = background_spikes_spectrogram
                    background_chunk['spikes_bands_spectrogram'] = background_bands_spectrogram
                    background_chunk['all_freqs'] = background_all_freqs
                    
                    # Save background chunk
                    background_chunk_dir = os.path.join(background_output_dir, f"sensor_{col_idx+1}")
                    os.makedirs(background_chunk_dir, exist_ok=True)
                    
                    background_chunk_file = os.path.join(background_chunk_dir, f"chunk_{col_idx+1}_data.pkl")
                    with open(background_chunk_file, 'wb') as f:
                        pickle.dump(background_chunk, f)
                    
                    # Create visualization
                    create_chunk_visualization(background_chunk, background_chunk_dir)
                    
                    background_chunks_processed += 1
                    print(f"✅ Background part of sensor {col_idx+1} processed and saved")
                    
                except Exception as e:
                    print(f"❌ Error processing background part of sensor {col_idx+1}: {str(e)}")
                    traceback.print_exc()
                
                # Clear memory after processing each sensor
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing sensor {col_idx+1}: {str(e)}")
                traceback.print_exc()
        
        # Create summary information
        summary = {
            'input_file': str(file_path),
            'human_end_time': human_end_time,
            'total_duration': 60.0,
            'sensor_count': column_count,
            'human_chunks_processed': human_chunks_processed,
            'background_chunks_processed': background_chunks_processed,
            'output_directories': {
                'human': human_output_dir,
                'background': background_output_dir
            }
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, "multi_sensor_processing_summary.pkl")
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        print("\n✅ MULTI-SENSOR FILE PROCESSING COMPLETE!")
        print(f"📊 Sensors processed: {column_count}")
        print(f"👤 Human chunks: {human_chunks_processed}")
        print(f"🔇 Background chunks: {background_chunks_processed}")
        print(f"📋 Summary saved to: {summary_file}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error processing multi-sensor file: {str(e)}")
        traceback.print_exc()
        return None

# Main execution function
def main():
    """
    Main execution function to demonstrate resonator processing
    """
    
    # Set paths for demonstration
    data_path = DATA_DIR
    
    # List of files to process
    files_to_process = []
    

     # Add other human file if enabled
    if PROCESS_HUMAN_FILE_OTHER:
        human_file = data_path / "40Sen_30Sec_stomping_30sec_quiet.csv"
        if human_file.exists():
            print(f"👣 Processing multi-sensor human file: {human_file}")
            process_multi_sensor_file(human_file)
            # Don't add to regular files_to_process as we handle it differently
            print("\n✅ Finished processing multi-sensor file!")
        else:
            print(f"❌ Multi-sensor human file not found: {human_file}")
            print(f"Looking in: {data_path}")
            # List available files in the data directory for debugging
            print("Available files in data directory:")
            try:
                for file in data_path.iterdir():
                    print(f"  - {file.name}")
            except Exception as e:
                print(f"Error listing directory: {e}")
    
    # Add car file 
    if PROCESS_CAR_FILE:
        car_file = data_path / "car.csv"
        if car_file.exists():
            files_to_process.append(car_file)
            print(f"🚗 Added car file: {car_file}")
        else:
            print(f"❌ Car file not found: {car_file}")
    
    # Add car_nothing file if enabled
    if PROCESS_CAR_NOTHING_FILE:
        car_nothing_file = data_path / "car_nothing.csv"
        if car_nothing_file.exists():
            files_to_process.append(car_nothing_file)
            print(f"🚙 Added car_nothing file: {car_nothing_file}")
        else:
            print(f"❌ Car nothing file not found: {car_nothing_file}")
    
    # Add human file if enabled
    if PROCESS_HUMAN_FILE:
        human_file = data_path / "human.csv"
        if human_file.exists():
            files_to_process.append(human_file)
            print(f"👣 Added human file: {human_file}")
        else:
            print(f"❌ Human file not found: {human_file}")

    # Add human_nothing file if enabled
    if PROCESS_HUMAN_NOTHING_FILE:
        human_nothing_file = data_path / "human_nothing.csv"
        if human_nothing_file.exists():
            files_to_process.append(human_nothing_file)
            print(f"👤 Added human_nothing file: {human_nothing_file}")
        else:
            print(f"❌ Human nothing file not found: {human_nothing_file}")
    
    # Process files
    if not files_to_process:
        print("❌ No files to process. Please check data paths.")
        return
    
    results = {}
    
    for file_path in files_to_process:
        print(f"\n📊 Processing file: {file_path}")
        result = visualize_resonator_output(file_path)
        results[str(file_path)] = result
    
    print("\n✅ Processing complete!")
    return results

# Execute main function when script is run directly
if __name__ == "__main__":
    main()
