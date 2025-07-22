#!/usr/bin/env python3
"""
Fixed Human Stepping File Processor
Handles 40-sensor geophone data correctly
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Manager
import warnings
warnings.filterwarnings('ignore')

# Import SCTN modules
sys.path.append('/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project')
from sctnN.spiking_network import SpikingNetwork
from sctnN.graphs import get_input_spikes_to

# Configuration
SAMPLING_FREQ = 1000  # Hz
TOTAL_DURATION = 60   # seconds
CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

# Resonator grids
clk_resonators_human = {
    153600: [22.1, 30.5, 33.9, 34.7, 41.2, 50.9, 52.6, 76.3, 63.6, 95.4]  # Human-optimized
}

clk_resonators_background = {
    153600: [15.2, 18.7, 22.1, 25.6, 29.1, 32.6, 36.1, 39.6, 43.1, 46.6, 50.1, 53.6, 57.1, 60.6, 64.1]  # Background-optimized
}

# Frequency bands for visualization
bands = {
    'LOW_FREQ': (15, 25),
    'CAR_APPROACH': (25, 35),
    'CAR_PEAK': (35, 45),
    'CAR_TAIL': (45, 55),
    'MID_GAP': (55, 65),
    'HUMAN_PEAK': (65, 75),
    'HUMAN_TAIL': (75, 85),
    'HIGH_FREQ': (85, 95)
}

def normalize_signal(signal):
    """Normalize signal to zero mean and unit variance"""
    if len(signal) == 0:
        return signal
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    return signal

def resample_signal(f_new, f_source, data):
    """Resample signal to new frequency"""
    if f_new == f_source:
        return data
    ratio = f_new / f_source
    new_length = int(len(data) * ratio)
    resampled = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
    return resampled

def process_single_resonator(f0, clk_freq, resampled_signal, progress_dict=None, resonator_id=None):
    """Process a single resonator with error handling"""
    try:
        # Create spiking network
        snn = SpikingNetwork()
        
        # Get input spikes
        input_spikes = get_input_spikes_to(snn, f0, clk_freq, resampled_signal)
        
        # Process through network
        output_spikes = snn.input(input_spikes)
        
        # Update progress if provided
        if progress_dict is not None and resonator_id is not None:
            progress_dict[resonator_id] = 1
        
        return {
            'f0': f0,
            'clk_freq': clk_freq,
            'input_spikes': input_spikes,
            'output_spikes': output_spikes
        }
        
    except Exception as e:
        print(f"❌ Error in resonator {f0}Hz: {e}")
        if progress_dict is not None and resonator_id is not None:
            progress_dict[resonator_id] = 1
        return None

def progress_monitor(progress_dict, total_resonators, resonator_weights, stop_event):
    """Monitor progress of resonator processing"""
    try:
        completed = 0
        while completed < total_resonators and not stop_event.is_set():
            completed = sum(progress_dict.values())
            if completed > 0:
                progress = completed / total_resonators
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                eta = "Complete!" if progress >= 1.0 else f"ETA: {int((1-progress)*30)}s"
                print(f"[{bar}] {progress*100:2.0f}% | {completed}/{total_resonators} | {eta}", end='\r')
            time.sleep(0.1)
        print()  # New line after progress
    except Exception as e:
        print(f"❌ Progress monitor error: {e}")

def process_with_resonator_grid_parallel(signal, fs, clk_resonators, duration, num_processes=None):
    """Process signal with resonator grid using parallel processing with error handling"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Limit to prevent memory issues
    
    print(f"Using {num_processes} processes for parallel computation")
    
    # Prepare resonators
    all_resonators = []
    resonator_weights = {}
    
    for clk_freq, frequencies in clk_resonators.items():
        print(f"Preparing resonators for clock frequency {clk_freq}")
        for f0 in frequencies:
            all_resonators.append((f0, clk_freq))
            resonator_weights[f0] = 1.0
    
    print(f"Processing {len(all_resonators)} resonators in parallel")
    
    # Resample signal
    resampled_signal = resample_signal(153600, fs, signal)
    
    # Process with error handling
    try:
        results = Parallel(n_jobs=num_processes, verbose=0, timeout=300)(
            delayed(process_single_resonator)(
                f0, clk_freq, resampled_signal
            ) for f0, clk_freq in all_resonators
        )
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) == 0:
            print("❌ No valid resonator results obtained")
            return {}
        
        # Organize results by clock frequency
        resonators_by_clk = {}
        for result in valid_results:
            clk_freq = result['clk_freq']
            if clk_freq not in resonators_by_clk:
                resonators_by_clk[clk_freq] = {}
            resonators_by_clk[clk_freq][result['f0']] = result
        
        print(f"✅ Successfully processed {len(valid_results)} resonators")
        return resonators_by_clk
        
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
        return {}

def spikes_event_spectrogram(clk_freq, events, window_ms=10, duration_s=None):
    """Create event spectrogram from spikes"""
    try:
        if not events or len(events) == 0:
            return np.zeros((1, 1))
        
        # Convert to time bins
        time_bins = int(duration_s * 1000 / window_ms) if duration_s else 1000
        spectrogram = np.zeros((1, time_bins))
        
        for event in events:
            time_ms = event * 1000 / clk_freq
            bin_idx = int(time_ms / window_ms)
            if 0 <= bin_idx < time_bins:
                spectrogram[0, bin_idx] += 1
        
        return spectrogram
        
    except Exception as e:
        print(f"❌ Error creating event spectrogram: {e}")
        return np.zeros((1, 1))

def events_to_max_spectrogram(resonators_by_clk, duration, clk_resonators, signal_file, main_clk=153600):
    """Convert resonator events to max spectrogram"""
    try:
        if main_clk not in resonators_by_clk:
            print(f"❌ No resonators found for clock frequency {main_clk}")
            return np.zeros((1, 1)), []
        
        all_freqs = []
        spectrograms = []
        
        for f0 in clk_resonators[main_clk]:
            if f0 in resonators_by_clk[main_clk]:
                result = resonators_by_clk[main_clk][f0]
                events = result.get('output_spikes', [])
                
                if events:
                    spec = spikes_event_spectrogram(main_clk, events, duration_s=duration)
                    spectrograms.append(spec)
                    all_freqs.append(f0)
        
        if not spectrograms:
            print("❌ No valid spectrograms created")
            return np.zeros((1, 1)), []
        
        # Stack and take max
        max_spectrogram = np.maximum.reduce(spectrograms)
        return max_spectrogram, all_freqs
        
    except Exception as e:
        print(f"❌ Error creating max spectrogram: {e}")
        return np.zeros((1, 1)), []

def spikes_to_bands(spectrogram, frequencies):
    """Convert frequency-based spectrogram to band-based"""
    try:
        if len(frequencies) == 0:
            return np.zeros((len(bands), spectrogram.shape[1]))
        
        band_spectrogram = np.zeros((len(bands), spectrogram.shape[1]))
        
        for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            # Find frequencies in this band
            band_freqs = [f for f in frequencies if fmin <= f <= fmax]
            if band_freqs:
                # Average the spectrogram for frequencies in this band
                band_spectrogram[i, :] = np.mean(spectrogram, axis=0)
        
        return band_spectrogram
        
    except Exception as e:
        print(f"❌ Error converting to bands: {e}")
        return np.zeros((len(bands), 1))

def load_and_split_human_file(file_path, human_end_time=32.5, sampling_freq=1000):
    """Load and split the 40-sensor geophone file correctly"""
    print(f"📂 Loading and splitting file: {file_path}")
    
    try:
        # Load the CSV file
        data = pd.read_csv(file_path, header=None)
        
        print(f"   📊 Loaded {len(data)} rows, {len(data.columns)} sensors")
        print(f"   📋 Columns found: {list(data.columns)}")
        
        # For multi-sensor data, we need to select a representative sensor
        # Let's use the first sensor (column 0) as the primary signal
        # You can modify this to use a different sensor or combine multiple sensors
        full_signal = data.iloc[:, 0].values  # Use first sensor
        
        # Clean the signal by removing NaN values
        nan_mask = np.isnan(full_signal)
        if nan_mask.any():
            print(f"   ⚠️  Found {nan_mask.sum()} NaN values, cleaning data...")
            # Find the last valid sample
            last_valid = np.where(~nan_mask)[0][-1] if (~nan_mask).any() else len(full_signal)
            full_signal = full_signal[:last_valid+1]
            print(f"   📊 After cleaning: {len(full_signal)} samples ({len(full_signal)/sampling_freq:.1f} seconds)")
        
        print(f"   📊 Using sensor 0: {len(full_signal)} samples ({len(full_signal)/sampling_freq:.1f} seconds)")
        
        # Verify we have the expected duration
        expected_samples = TOTAL_DURATION * sampling_freq
        if len(full_signal) != expected_samples:
            print(f"   ⚠️  Warning: Expected {expected_samples} samples, got {len(full_signal)}")
            # Truncate or pad to expected length
            if len(full_signal) > expected_samples:
                full_signal = full_signal[:expected_samples]
                print(f"   📊 Truncated to {len(full_signal)} samples")
            elif len(full_signal) < expected_samples:
                # Pad with zeros
                padding = np.zeros(expected_samples - len(full_signal))
                full_signal = np.concatenate([full_signal, padding])
                print(f"   📊 Padded to {len(full_signal)} samples")
        
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
            vmax = np.percentile(spikes_bands_spectrogram, 97) if spikes_bands_spectrogram.size > 0 else 1
        else:
            vmax = np.percentile(spikes_bands_spectrogram, 99) if spikes_bands_spectrogram.size > 0 else 1
        
        if spikes_bands_spectrogram.size > 0:
            im = axs[1].imshow(spikes_bands_spectrogram, aspect='auto', cmap='jet', origin='lower',
                              extent=[0, duration, 0, len(bands)], vmin=0, vmax=vmax)
            axs[1].set_yticks(np.arange(len(band_labels)) + 0.5)
            axs[1].set_yticklabels(band_labels)
            fig.colorbar(im, ax=axs[1], label='Spike Activity', pad=0.01)
        else:
            axs[1].text(0.5, 0.5, 'No spike data available', ha='center', va='center', transform=axs[1].transAxes)
        
        axs[1].set_title(f'{section_name} - Chunk {chunk_idx} - Resonator Spikegram', fontsize=14)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Frequency Band')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def process_signal_section(signal, section_name, output_dir, chunk_duration=30, num_processes=8):
    """Process a signal section using resonator pipeline with error handling"""
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
            print(f"⚠️  Skipping empty chunk {chunk_idx}")
            continue
        
        try:
            # Process with resonators
            print(f"⚡ Processing chunk {chunk_idx} with resonator grid...")
            resonator_output = process_with_resonator_grid_parallel(
                chunk_signal, SAMPLING_FREQ, clk_resonators, current_chunk_duration, num_processes
            )
            
            if not resonator_output:
                print(f"⚠️  No resonator output for chunk {chunk_idx}, skipping")
                continue
            
            # Create spectrograms
            print(f"📊 Creating spike spectrograms for chunk {chunk_idx}...")
            max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
                resonator_output, current_chunk_duration, clk_resonators, section_name
            )
            
            if max_spikes_spectrogram.size == 0:
                print(f"⚠️  No spectrogram data for chunk {chunk_idx}, skipping")
                continue
            
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

def process_human_stepping_file(file_path, human_end_time=32.5, chunk_duration=30, num_processes=8):
    """Main function to process the human stepping file"""
    print("🧠 HUMAN STEPPING FILE PROCESSOR (FIXED)")
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
        print("Usage: python process_human_file_fixed.py <path_to_csv_file>")
        print("Example: python process_human_file_fixed.py /path/to/40Sen_30Sec_stomping_30sec_quiet.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    print("🧠 FIXED HUMAN STEPPING RESONATOR PROCESSOR")
    print("=" * 80)
    print("Processing 60-second 40-sensor geophone file:")
    print("  • 32.5 seconds of human stepping (0-32.5s)")
    print("  • 27.5 seconds of background/no stepping (32.5-60s)")
    print("  • Using sensor 0 from 40-sensor array")
    print("=" * 80)
    
    # Process the file
    summary = process_human_stepping_file(
        file_path=file_path,
        human_end_time=32.5,
        chunk_duration=30,
        num_processes=8  # Reduced to prevent memory issues
    )
    
    if summary:
        print("\n🎉 SUCCESS! File processed successfully.")
        print(f"Check the output directories:")
        print(f"  👤 Human: {summary['output_directories']['human']}")
        print(f"  🔇 Background: {summary['output_directories']['background']}")
    else:
        print("\n❌ Processing failed. Check error messages above.") 