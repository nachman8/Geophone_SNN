#!/usr/bin/env python3
"""
Multi-sensor geophone processor: processes each sensor (column) as a separate chunk using the exact pipeline from Geophone_Model_SNN.py.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path

# Import the main pipeline functions from Geophone_Model_SNN.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress the header printing from Geophone_Model_SNN.py when importing
import contextlib
import io

# Temporarily redirect stdout to suppress header printing during import
with contextlib.redirect_stdout(io.StringIO()):
    from MyCode.Geophone_Model_SNN import (
        normalize_signal,
        process_with_resonator_grid_parallel,
        events_to_max_spectrogram,
        spikes_to_bands,
        create_chunk_visualization,
        clk_resonators_human,
        clk_resonators_car,  # <-- fix import here
        CHUNKED_OUTPUT_DIR,
        bands
    )

SAMPLING_FREQ = 1000  # Hz
HUMAN_END_TIME = 32.5
TOTAL_DURATION = 60.0
CHUNKED_OUTPUT_HUMAN = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output/human_new"
CHUNKED_OUTPUT_BG = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output/human_nothing_new"

DATA_FILE = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data/40Sen_30Sec_stomping_30sec_quiet.csv"

os.makedirs(CHUNKED_OUTPUT_HUMAN, exist_ok=True)
os.makedirs(CHUNKED_OUTPUT_BG, exist_ok=True)

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
    # Split
    split_sample = int(HUMAN_END_TIME * sampling_freq)
    human_signal = normalize_signal(signal[:split_sample])
    background_signal = normalize_signal(signal[split_sample:])
    # Also return time arrays for both
    time_human = np.arange(len(human_signal)) / sampling_freq
    time_background = np.arange(len(background_signal)) / sampling_freq
    return human_signal, background_signal, time_human, time_background

def process_and_save(signal, section_name, chunk_idx, output_dir, clk_resonators, duration, num_processes=8, time_arr=None):
    chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx}")
    os.makedirs(chunk_dir, exist_ok=True)
    print(f"   → Processing {section_name} chunk_{chunk_idx} ({duration:.1f}s)...")
    
    # Clear any existing progress monitoring
    import gc
    gc.collect()
    
    # Resonator processing
    resonator_output = process_with_resonator_grid_parallel(
        signal, SAMPLING_FREQ, clk_resonators, duration, num_processes=num_processes
    )
    
    # Clear progress monitoring after processing
    gc.collect()
    
    # Spikegram
    max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
        resonator_output, duration, clk_resonators, section_name
    )
    spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)
    # Save
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
    viz_path = os.path.join(chunk_dir, f"{section_name}_chunk_{chunk_idx}_visualization.png")
    create_chunk_visualization(chunk_result, chunk_dir)
    print(f"   ✓ Saved chunk_{chunk_idx} to {chunk_file}")
    
    # Clear memory after processing
    del resonator_output, max_spikes_spectrogram, spikes_bands_spectrogram, chunk_result
    gc.collect()

def main():
    print("🧠 MULTI-SENSOR GEOPHONE PROCESSOR (EXACT PIPELINE)")
    print("=" * 80)
    print(f"Processing file: {DATA_FILE}")
    print("Each sensor (column) will be processed as a separate chunk.")
    print("Output directories:")
    print(f"  {CHUNKED_OUTPUT_HUMAN}")
    print(f"  {CHUNKED_OUTPUT_BG}")
    print("=" * 80)
    data = pd.read_csv(DATA_FILE, header=None)
    n_sensors = data.shape[1]
    for sensor_idx in range(n_sensors):
        human_signal, background_signal, time_human, time_background = process_sensor_column(sensor_idx, data, SAMPLING_FREQ)
        # Human section
        process_and_save(
            human_signal, "human_new", sensor_idx, CHUNKED_OUTPUT_HUMAN, clk_resonators_human, HUMAN_END_TIME, num_processes=8, time_arr=time_human
        )
        # Background section (use car grid for background)
        process_and_save(
            background_signal, "human_nothing_new", sensor_idx, CHUNKED_OUTPUT_BG, clk_resonators_car, TOTAL_DURATION-HUMAN_END_TIME, num_processes=8, time_arr=time_background
        )
    print("\n✅ All sensors processed. Check output directories for results.")

if __name__ == "__main__":
    main() 