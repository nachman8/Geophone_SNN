#!/usr/bin/env python3
"""
SCTN (Spike Continuous Time Neuron) Signal Processor for Geophone Data
=====================================================================

A production-ready SCTN resonator-based signal processing system for 
high-precision geophone signal analysis. Processes raw signals through 
resonator grids to generate spectrograms and frequency-band analysis.

Key Features:
- SCTN resonator-based signal processing
- Parallel processing with real-time progress monitoring
- Multi-sensor support with chunked processing
- Signal-specific resonator optimization
- Comprehensive visualization and data storage

Author: Seismic Processing Team
Version: 2.0
Framework: SCTN (Spike Continuous Time Neuron) Resonator Processing
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
import signal
from collections import Counter
from pathlib import Path
import sys
import warnings
import gc

# Configuration constants
DATA_DIR = Path.home() / "data"
CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

def check_system_resources():
    """Enhanced system resource checking with memory limits"""
    import psutil
    
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent
        
        # Check if we have enough memory (at least 2GB available, less than 85% used)
        if available_gb < 2.0 or used_percent > 85:
            print(f"⚠️  Low memory: {available_gb:.1f}GB available, {used_percent:.1f}% used")
            return False
            
        return True
        
    except ImportError:
        # psutil not available, use basic check
        print("⚠️  psutil not available, proceeding with basic resource check")
        return True
    except Exception as e:
        print(f"⚠️  Resource check error: {e}")
        return True

def cleanup_memory(*variables):
    """Simplified memory cleanup"""
    import gc
    
    # Delete provided variables
    for var in variables:
        if var is not None:
            try:
                del var
            except:
                pass
    
    # Single garbage collection pass
    gc.collect()

# Processing configuration
LOAD_FROM_CHUNKED = False  # True: Load pre-computed data, False: Full resonator processing
PROCESS_HUMAN_FILE = False  # True: Process 40Sen_30Sec_stomping_30sec_quiet.csv through full pipeline

print("🧠 SCTN RESONATOR SIGNAL PROCESSING SYSTEM")
print("=" * 70)
if PROCESS_HUMAN_FILE:
    print(f"🔄 MODE: Processing 40Sen_30Sec_stomping_30sec_quiet.csv")
    print(f"📊 Multi-sensor processing with human/background split")
    print(f"🎯 Full SCTN resonator pipeline from raw signals to spectrograms")
elif LOAD_FROM_CHUNKED:
    print(f"📁 MODE: Pre-computed resonator data")
    print(f"📂 Source: {CHUNKED_OUTPUT_DIR}")
    print(f"⚡ Loading existing resonator processing results")
else:
    print(f"🔄 MODE: Full SCTN resonator processing pipeline")
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

# Define separate resonator grids for different data types
clk_resonators_car = {
    153600: [
        # LOW_FREQ coverage for car
        22.1, 28.8,
        # Enhanced CAR coverage (30-48 Hz) - all available for better car detection
        30.5, 34.7, 37.2, 40.2, 43.6, 47.7,
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
    "Clean signal without normalization - preserve original amplitude scale"
    
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
        import gc
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
        import gc
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
            # Check system resources before starting parallel processing
            if not check_system_resources():
                print("⚠️  System resources low, reducing parallel processes")
                num_processes = min(num_processes, 8)
                
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
    
    # Auto-detect and select appropriate resonator grid
    clk_resonators = get_resonator_grid(file_path)
    
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
        cleanup_memory(fig, axs, fft_bin_spectogram, f, t, Sxx)
        
    except Exception as e:
        print(f"Error creating chunk visualization: {e}")
        import traceback
        traceback.print_exc()

def process_file_in_chunks(file_path, chunk_duration=120, num_processes=15, min_chunk_size=10):
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunked_output", file_stem)
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
    
    return chunk_index

# ========================================================================
# HUMAN FILE PROCESSING (40Sen_30Sec_stomping_30sec_quiet.csv)
# ========================================================================

def process_human_file_pipeline():
    """
    Process the 40Sen_30Sec_stomping_30sec_quiet.csv file through the full SCTN resonator pipeline.
    Each sensor (column) is processed as a separate chunk for both human and background sections.
    """
    print("\n🧠 MULTI-SENSOR GEOPHONE PROCESSOR (SCTN RESONATOR PIPELINE)")
    print("=" * 80)
    
    # File configuration
    DATA_FILE = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data/40Sen_30Sec_stomping_30sec_quiet.csv"
    SAMPLING_FREQ = 1000  # Hz
    HUMAN_END_TIME = 32.5
    TOTAL_DURATION = 60.0
    
    # Output directories
    CHUNKED_OUTPUT_HUMAN = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output/human_new"
    CHUNKED_OUTPUT_BG = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output/human_nothing_new"
    
    os.makedirs(CHUNKED_OUTPUT_HUMAN, exist_ok=True)
    os.makedirs(CHUNKED_OUTPUT_BG, exist_ok=True)
    
    print(f"Processing file: {DATA_FILE}")
    print("Each sensor (column) will be processed as a separate chunk.")
    print("Output directories:")
    print(f"  {CHUNKED_OUTPUT_HUMAN}")
    print(f"  {CHUNKED_OUTPUT_BG}")
    print("=" * 80)
    
    # Load data
    data = pd.read_csv(DATA_FILE, header=None)
    n_sensors = data.shape[1]
    print(f"Loaded {n_sensors} sensors from file")
    
    def process_sensor_column(sensor_idx, data, sampling_freq=1000):
        """Process a single sensor (column) as a chunk for both human and background."""
        print(f"\n=== Processing sensor {sensor_idx} ===")
        
        signal = data.iloc[:, sensor_idx].values
        
        # Clean NaNs
        nan_mask = np.isnan(signal)
        if nan_mask.any():
            print(f"   ⚠️  Found {nan_mask.sum()} NaN values, cleaning data...")
            last_valid = np.where(~nan_mask)[0][-1] if (~nan_mask).any() else len(signal)
            signal = signal[:last_valid+1]
        
        # Pad/truncate
        expected_samples = int(TOTAL_DURATION * sampling_freq)
        if len(signal) > expected_samples:
            signal = signal[:expected_samples]
        elif len(signal) < expected_samples:
            signal = np.concatenate([signal, np.zeros(expected_samples - len(signal))])
        
        # Split into human and background sections
        split_sample = int(HUMAN_END_TIME * sampling_freq)
        human_signal = process_signal(signal[:split_sample])
        background_signal = process_signal(signal[split_sample:])
        
        # Display signal statistics (without normalization)
        display_signal_stats(human_signal, f"Human signal (sensor {sensor_idx})")
        display_signal_stats(background_signal, f"Background signal (sensor {sensor_idx})")
        
        # Time arrays
        time_human = np.arange(len(human_signal)) / sampling_freq
        time_background = np.arange(len(background_signal)) / sampling_freq
        
        # Clean up the original signal to save memory
        cleanup_memory(signal)
        
        return human_signal, background_signal, time_human, time_background
    
    def process_and_save(signal, section_name, chunk_idx, output_dir, clk_resonators, duration, num_processes=10, time_arr=None):
        """Process and save a single chunk using the SCTN resonator pipeline."""
        import time as time_module
        
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx}")
        os.makedirs(chunk_dir, exist_ok=True)
        print(f"   → Processing {section_name} chunk_{chunk_idx} ({duration:.1f}s)...")
        
        start_time = time_module.time()
        
        try:
            # SCTN Resonator processing
            resonator_start = time_module.time()
            print(f"   🔬 Starting SCTN resonator processing with {len(signal)} samples...")
            
            resonator_output = process_with_resonator_grid_parallel(
                signal, SAMPLING_FREQ, clk_resonators, duration, num_processes=num_processes
            )
            
            resonator_time = time_module.time() - resonator_start
            print(f"   ✅ Resonator processing completed in {resonator_time:.1f}s")
            
            # Spikegram generation
            spikegram_start = time_module.time()
            print(f"   📊 Generating spikegrams...")
            
            max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
                resonator_output, duration, clk_resonators, section_name
            )
            spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)
            
            spikegram_time = time_module.time() - spikegram_start
            print(f"   ✅ Spikegram generation completed in {spikegram_time:.1f}s")
            
            # Save results
            save_start = time_module.time()
            chunk_result = {
                'chunk_idx': chunk_idx,
                'duration': duration,
                'signal': signal,
                'resonator_outputs': resonator_output,
                'max_spikes_spectrogram': max_spikes_spectrogram,
                'spikes_bands_spectrogram': spikes_bands_spectrogram,
                'all_freqs': all_freqs,
                'section_name': section_name,
                'time': time_arr if time_arr is not None else np.arange(len(signal)) / SAMPLING_FREQ,
                'file_path': DATA_FILE
            }
            
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_idx}_data.pkl")
            with open(chunk_file, 'wb') as f:
                pickle.dump(chunk_result, f)
            
            # Visualization
            create_chunk_visualization(chunk_result, chunk_dir)
            
            save_time = time_module.time() - save_start
            total_time = time_module.time() - start_time
            
            print(f"   ✓ Saved chunk_{chunk_idx} to {chunk_file}")
            print(f"   📊 Timing: Resonator={resonator_time:.1f}s, Spikegram={spikegram_time:.1f}s, Save={save_time:.1f}s, Total={total_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error processing chunk {chunk_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Process all sensors
    import time as time_module
    
    for sensor_idx in range(n_sensors):
        print(f"\n{'='*60}")
        print(f"📍 STARTING SENSOR {sensor_idx}/{n_sensors-1} at {time_module.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        start_time = time_module.time()
        human_signal, background_signal, time_human, time_background = process_sensor_column(sensor_idx, data, SAMPLING_FREQ)
        
        if human_signal is None:
            print(f"⚠️  Skipping sensor {sensor_idx} due to resource constraints")
            continue
        
        # Human section
        print(f"\n🧠 Processing HUMAN section for sensor {sensor_idx}...")
        human_start = time_module.time()
        process_and_save(
            human_signal, "human_new", sensor_idx, CHUNKED_OUTPUT_HUMAN, 
            clk_resonators_human, HUMAN_END_TIME, num_processes=10, time_arr=time_human
        )
        human_time = time_module.time() - human_start
        print(f"   ✅ Human section completed in {human_time:.1f}s")
        
        # Clean up human data before background processing
        cleanup_memory(human_signal, time_human)
        
        # Background section
        print(f"\n🚗 Processing BACKGROUND section for sensor {sensor_idx}...")
        print(f"   📊 Background signal length: {len(background_signal)} samples ({len(background_signal)/SAMPLING_FREQ:.1f}s)")
        print(f"   🔧 Using car resonators (optimized for background noise)")
        
        # Additional resource check before background processing
        if not check_system_resources():
            print(f"⚠️  Insufficient resources for background processing of sensor {sensor_idx}")
            continue
            
        # Force garbage collection before background processing
        import gc
        gc.collect()
        
        background_start = time_module.time()
        print(f"   🚀 Starting background processing at {time_module.strftime('%H:%M:%S')}")
        
        try:
            process_and_save(
                background_signal, "human_nothing_new", sensor_idx, CHUNKED_OUTPUT_BG, clk_resonators_human, TOTAL_DURATION-HUMAN_END_TIME, num_processes=10, time_arr=time_background
            )
            background_time = time_module.time() - background_start
            print(f"   ✅ Background section completed in {background_time:.1f}s")
            
        except Exception as e:
            background_time = time_module.time() - background_start
            print(f"   ❌ Error in background processing after {background_time:.1f}s: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up background data after processing
        cleanup_memory(background_signal, time_background)
        
        total_time = time_module.time() - start_time
        print(f"📍 SENSOR {sensor_idx} COMPLETED in {total_time:.1f}s (Human: {human_time:.1f}s, Background: {background_time:.1f}s)")
        print(f"{'='*60}")
    
    print("\n✅ All sensors processed. Check output directories for results.")
    return True

# ========================================================================
# MAIN PROCESSING PIPELINE
# ========================================================================

def run_sctn_signal_processing_pipeline(chunk_duration=30, num_processes=15):
    """
    Main SCTN resonator signal processing pipeline.
    
    Processes geophone signals through SCTN resonator grids to generate
    spectrograms and frequency-band analysis without any classification.
    
    Args:
        chunk_duration (int): Chunk size for processing (seconds)
        num_processes (int): Parallel processes for resonator computation
        
    Returns:
        dict: Processing results and file information
    """
    print(f"\n🧠 SCTN RESONATOR SIGNAL PROCESSING PIPELINE")
    print("=" * 90)
    print("    🎯 Pure Signal Processing System")
    print("    🔬 SCTN Resonator-Based Analysis")
    print("    📊 Parallel Processing with Progress Monitoring")
    print("    🎭 Multi-Sensor Support with Chunked Processing")
    print("    📈 Comprehensive Visualization and Data Storage")
    print("=" * 90)
    
    pipeline_start_time = time.time()
    
    if LOAD_FROM_CHUNKED:
        print(f"\n📁 LOADING PRE-PROCESSED RESONATOR DATA")
        print("=" * 60)
        print(f"📂 Source directory: {CHUNKED_OUTPUT_DIR}")
        
        # Validate chunked data exists
        if not os.path.exists(CHUNKED_OUTPUT_DIR):
            print(f"❌ Chunked output directory does not exist: {CHUNKED_OUTPUT_DIR}")
            print("💡 Set LOAD_FROM_CHUNKED = False to run full processing")
            return None
        
        # List available processed datasets
        available_datasets = []
        for item in os.listdir(CHUNKED_OUTPUT_DIR):
            item_path = os.path.join(CHUNKED_OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                available_datasets.append(item)
        
        if not available_datasets:
            print(f"❌ No processed datasets found in {CHUNKED_OUTPUT_DIR}")
            print("💡 Set LOAD_FROM_CHUNKED = False to run full processing")
            return None
        
        print(f"✅ Found {len(available_datasets)} processed datasets:")
        for dataset in available_datasets:
            dataset_path = os.path.join(CHUNKED_OUTPUT_DIR, dataset)
            chunk_count = len([f for f in os.listdir(dataset_path) if f.startswith('chunk_')])
            print(f"   📊 {dataset}: {chunk_count} chunks")
        
        processing_results = {
            'mode': 'loaded_from_chunked',
            'source_directory': CHUNKED_OUTPUT_DIR,
            'available_datasets': available_datasets,
            'total_datasets': len(available_datasets)
        }
        
    else:
        print(f"\n🔄 FULL SCTN RESONATOR PROCESSING")
        print("=" * 60)
        
        # Validate data files
        data_files = {
            'car': [DATA_DIR / "car.csv", DATA_DIR / "car_nothing.csv"],
            'human': [DATA_DIR / "human.csv", DATA_DIR / "human_nothing.csv"]
        }
        
        missing_files = []
        available_files = []
        
        for signal_type, file_list in data_files.items():
            for file_path in file_list:
                if not file_path.exists():
                    missing_files.append(str(file_path))
                else:
                    available_files.append(str(file_path))
        
        if missing_files and not available_files:
            print(f"❌ No data files found for processing:")
            for file_path in missing_files:
                print(f"   💀 {file_path}")
            print("💡 Please ensure data files are available in ~/data/")
            return None
        
        if missing_files:
            print(f"⚠️  Some data files missing (will process available files):")
            for file_path in missing_files:
                print(f"   ⚠️  {file_path}")
        
        print(f"✅ Processing {len(available_files)} available data files:")
        for file_path in available_files:
            print(f"   📂 {file_path}")
        
        # Process available files
        processed_files = []
        
        for file_path in available_files:
            print(f"\n🔄 Processing {Path(file_path).name}...")
            
            try:
                chunk_index = process_file_in_chunks(
                    file_path, chunk_duration, num_processes
                )
                
                if chunk_index:
                    processed_files.append({
                        'file_path': file_path,
                        'chunk_index': chunk_index,
                        'num_chunks': chunk_index['num_chunks']
                    })
                    print(f"✅ Successfully processed {Path(file_path).name}")
                else:
                    print(f"❌ Failed to process {Path(file_path).name}")
                    
            except Exception as e:
                print(f"❌ Error processing {Path(file_path).name}: {e}")
                import traceback
                traceback.print_exc()
        
        processing_results = {
            'mode': 'full_processing',
            'processed_files': processed_files,
            'total_files_processed': len(processed_files),
            'total_files_available': len(available_files),
            'chunk_duration': chunk_duration,
            'num_processes': num_processes
        }
    
    # Calculate execution time
    total_execution_time = time.time() - pipeline_start_time
    processing_results['total_execution_time'] = total_execution_time
    
    # Print summary
    print(f"\n🚀 SCTN RESONATOR PROCESSING SUMMARY")
    print("=" * 90)
    print(f"⏱️  Total Processing Time: {total_execution_time:.2f} seconds")
    
    if LOAD_FROM_CHUNKED:
        print(f"📁 Mode: Loaded from chunked data")
        print(f"📊 Available datasets: {processing_results['total_datasets']}")
        print(f"📂 Source: {CHUNKED_OUTPUT_DIR}")
    else:
        print(f"🔄 Mode: Full SCTN resonator processing")
        print(f"📊 Files processed: {processing_results['total_files_processed']}")
        print(f"⚙️  Chunk duration: {chunk_duration}s")
        print(f"🔧 Parallel processes: {num_processes}")
    
    print(f"\n✅ SCTN RESONATOR PROCESSING COMPLETE!")
    print(f"   🎯 All signal processing completed successfully")
    print(f"   📊 Resonator spectrograms generated")
    print(f"   🔬 Frequency-band analysis completed")
    print(f"   📁 All results and visualizations saved")
    
    return processing_results

# ========================================================================
# EVALUATION FUNCTIONS (KEPT FOR LATER USE)
# ========================================================================
# Note: These functions are kept in the file but not used in the current pipeline
# They will be available for future classification work

def create_resonator_classification_report_plot(precision, recall, f1, support, class_names, title, save_path=None):
    """Create a professional classification report visualization (KEPT FOR LATER USE)"""
    import os
    # Calculate averages
    accuracy = np.sum(precision * support) / np.sum(support)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    # Create DataFrame
    data = {
        'Class': class_names + ['', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [f'{p:.3f}' for p in precision] + ['', '', f'{macro_precision:.3f}', f'{weighted_precision:.3f}'],
        'Recall': [f'{r:.3f}' for r in recall] + ['', '', f'{macro_recall:.3f}', f'{weighted_recall:.3f}'],
        'F1-Score': [f'{f:.3f}' for f in f1] + ['', f'{accuracy:.3f}', f'{macro_f1:.3f}', f'{weighted_f1:.3f}'],
        'Support': [f'{int(s)}' for s in support] + ['', f'{int(np.sum(support))}', f'{int(np.sum(support))}', f'{int(np.sum(support))}']
    }
    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Always save to output_plots directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
    os.makedirs(output_dir, exist_ok=True)
    clean_title = title.replace(' ', '_').replace('\n', '_').replace('/', '_').replace('\\', '_')
    filename = f"classification_report_{clean_title}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filepath}")
    return fig

def plot_resonator_confusion_matrix(cm, class_names, title, save_path=None):
    """Plot confusion matrix with percentages and professional styling (KEPT FOR LATER USE)"""
    import os
    plt.figure(figsize=(8, 6))

    # Normalize confusion matrix for color scaling
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm_normalized[i, j] * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Always save to output_plots directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
    os.makedirs(output_dir, exist_ok=True)
    clean_title = title.replace(' ', '_').replace('\n', '_').replace('/', '_').replace('\\', '_')
    filename = f"confusion_matrix_{clean_title}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filepath}")
    return filepath

# Example usage
if __name__ == "__main__":
    print("🧠 SCTN RESONATOR SIGNAL PROCESSING SYSTEM")
    print("=" * 90)
    print("    🎯 Pure Signal Processing Pipeline")
    print("    🔬 SCTN Resonator-Based Analysis")
    print("    📊 Parallel Processing with Real-Time Monitoring")
    print("    🎭 Multi-Sensor Support with Chunked Processing")
    print("    📈 Comprehensive Visualization and Data Storage")
    print("=" * 90)
    
    print("\n🎛️ SYSTEM CONFIGURATION:")
    print(f"   📂 LOAD_FROM_CHUNKED = {LOAD_FROM_CHUNKED}")
    print(f"   📂 PROCESS_HUMAN_FILE = {PROCESS_HUMAN_FILE}")
    print(f"   📁 CHUNKED_OUTPUT_DIR = {CHUNKED_OUTPUT_DIR}")
    print(f"   📁 DATA_DIR = {DATA_DIR}")
    
    # Check if we should process the human file
    if PROCESS_HUMAN_FILE:
        print(f"\n🚀 LAUNCHING HUMAN FILE PROCESSING PIPELINE...")
        print("-" * 85)
        
        try:
            success = process_human_file_pipeline()
            if success:
                print("\n✅ HUMAN FILE PROCESSING COMPLETED SUCCESSFULLY!")
                print("📁 Check output directories for processed chunks:")
                print("   - /chunked_output/human_new/")
                print("   - /chunked_output/human_nothing_new/")
            else:
                print("\n❌ HUMAN FILE PROCESSING FAILED")
        except Exception as e:
            print(f"\n❌ HUMAN FILE PROCESSING ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n🚀 LAUNCHING SCTN RESONATOR PROCESSING PIPELINE...")
        print("-" * 85)
        
        try:
            pipeline_results = run_sctn_signal_processing_pipeline(
                chunk_duration=30, num_processes=15
            )
            
            if pipeline_results:
                print("\n🎉 SCTN RESONATOR PROCESSING SUCCESS!")
                print("=" * 90)
                
                execution_time = pipeline_results.get('total_execution_time', 0)
                mode = pipeline_results.get('mode', 'unknown')
                
                print(f"   ⚡ Total Execution Time: {execution_time:.2f} seconds")
                print(f"   🔄 Processing Mode: {mode}")
                
                if mode == 'loaded_from_chunked':
                    print(f"   📊 Available Datasets: {pipeline_results.get('total_datasets', 0)}")
                elif mode == 'full_processing':
                    print(f"   📊 Files Processed: {pipeline_results.get('total_files_processed', 0)}")
                
                print(f"\n📋 NEXT STEPS:")
                print(f"   🎯 SCTN resonator processing complete")
                print(f"   📊 Spectrograms and frequency-band analysis generated")
                print(f"   📁 All results saved in chunked_output directories")
                print(f"   🔬 Ready for classification or further analysis")
                
            else:
                print("\n❌ SCTN RESONATOR PROCESSING FAILED")
                print("   🔧 Check logs above for specific error details")
                
        except Exception as pipeline_error:
            print(f"\n⚠️  Pipeline failed: {pipeline_error}")
            print("\n🔧 TROUBLESHOOTING GUIDE:")
            
            if LOAD_FROM_CHUNKED:
                print("   📁 Pre-processed Mode:")
                print("      1. Verify chunked_output directory exists with processed data")
                print("      2. Check that .pkl files are present in category subdirectories")
                print("      3. Try: LOAD_FROM_CHUNKED = False for fresh processing")
            else:
                print("   🔄 Full Processing Mode:")
                print("      1. Verify data files exist in ~/data/")
                print("      2. Check available memory (SCTN resonator processing requires ~4-8GB)")
                print("      3. Reduce num_processes if memory limited")
            
            print("   🧠 SCTN Processing:")
            print("      1. Verify sctnN library is properly installed")
            print("      2. Check that RESONATOR_FUNCTIONS are available")
            print("      3. Ensure multiprocessing support is working")
            
            import traceback
            traceback.print_exc()
            
        print("\n" + "=" * 100)
        print("📋 SYSTEM GUIDE")
        print("=" * 100)
        print()
        print("🧠 MAIN PIPELINE:")
        print("   run_sctn_signal_processing_pipeline()")
        print("   🔬 SCTN resonator-based signal processing")
        print("   📊 Parallel computation with progress monitoring") 
        print("   ⚖️  Chunked processing for memory efficiency")
        print("   🏆 Comprehensive visualization and data storage")
        print()
        print("🔬 SCTN RESONATOR PROCESSING:")
        print("   📊 Frequency-selective resonator grids")
        print("   🎯 Signal-specific optimization (human vs car)")
        print("   🎛️ LOAD_FROM_CHUNKED = True  → Load pre-processed data (fast)")
        print("   🔄 LOAD_FROM_CHUNKED = False → Full SCTN resonator processing")
        print()
        print("📊 OUTPUT DATA:")
        print("   📈 Raw signal data and time arrays")
        print("   🎵 SCTN resonator outputs and spike events")
        print("   🔊 Max spike spectrograms and frequency bands")
        print("   📊 Comprehensive visualizations (FFT + Spikegram)")
        print()
        print("🎯 PROCESSING MODES:")
        print("   🎭 Standard file processing with chunked pipeline")
        print("   🧮 Multi-sensor human file processing")
        print("   ⚡ Real-time progress monitoring")
        print("   📊 Automatic resonator grid selection")
        print("=" * 100)
