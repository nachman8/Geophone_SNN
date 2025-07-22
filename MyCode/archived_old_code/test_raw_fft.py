#!/usr/bin/env python3
"""
Test Raw Data FFT Processing
============================

This script tests the raw data loading and FFT feature extraction
to ensure the pipeline works correctly.
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from pathlib import Path

RAW_DATA_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data"
CHUNK_DURATION = 30  # seconds
SAMPLING_FREQ = 1000  # Hz

def extract_fft_features(signal, fs=1000):
    """Extract comprehensive FFT-based features from signal"""
    try:
        # FFT analysis
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), 1/fs)
        
        # Power spectral density
        power = np.abs(fft_vals)**2
        
        # Focus on positive frequencies up to 100 Hz (seismic range)
        max_freq_idx = int(100 * len(freqs) / fs)
        freqs_pos = freqs[:max_freq_idx]
        power_pos = power[:max_freq_idx]
        
        # Frequency band features
        bands = {
            'low_freq': (1, 10),
            'car_low': (10, 25),
            'car_mid': (25, 40),
            'car_high': (40, 60),
            'human_low': (60, 80),
            'human_high': (80, 100)
        }
        
        features = []
        
        # Band power features
        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs_pos >= f_low) & (freqs_pos <= f_high)
            band_power = np.sum(power_pos[band_mask])
            features.append(band_power)
        
        # Spectral statistics
        features.extend([
            np.mean(power_pos),           # Mean power
            np.std(power_pos),            # Power variability
            np.max(power_pos),            # Peak power
            np.argmax(power_pos),         # Dominant frequency index
            np.sum(power_pos),            # Total power
        ])
        
        # Spectral centroid and bandwidth
        spectral_centroid = np.sum(freqs_pos * power_pos) / np.sum(power_pos)
        spectral_bandwidth = np.sqrt(np.sum(((freqs_pos - spectral_centroid)**2) * power_pos) / np.sum(power_pos))
        
        features.extend([spectral_centroid, spectral_bandwidth])
        
        # Time domain features
        features.extend([
            np.mean(signal),              # Mean amplitude
            np.std(signal),               # Standard deviation
            np.max(signal),               # Peak amplitude
            np.min(signal),               # Minimum amplitude
            np.sqrt(np.mean(signal**2)),  # RMS
        ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting FFT features: {e}")
        return None

def test_raw_fft_processing():
    """Test raw data FFT processing"""
    print("🧪 TESTING RAW DATA FFT PROCESSING")
    print("=" * 50)
    
    raw_features = []
    raw_labels = []
    
    files = {
        'human': Path(RAW_DATA_DIR) / 'human.csv',
        'human_nothing': Path(RAW_DATA_DIR) / 'human_nothing.csv',
        'car': Path(RAW_DATA_DIR) / 'car.csv',
        'car_nothing': Path(RAW_DATA_DIR) / 'car_nothing.csv'
    }
    
    for category, file_path in files.items():
        print(f"\n📂 Testing {category}...")
        
        if file_path.exists():
            # Load a small sample to test
            print(f"  Loading file: {file_path}")
            df = pd.read_csv(file_path, nrows=90000)  # 90 seconds of data for testing
            signal = df['amplitude'].values
            
            print(f"  Original signal: {len(signal)} samples")
            
            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Split into 30-second chunks
            chunk_size = CHUNK_DURATION * SAMPLING_FREQ  # 30 seconds * 1000 Hz
            num_chunks = len(signal) // chunk_size
            
            print(f"  Creating {num_chunks} chunks of {chunk_size} samples each")
            
            category_features = []
            
            for i in range(min(num_chunks, 3)):  # Test first 3 chunks
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk_signal = signal[start_idx:end_idx]
                
                # Extract FFT features
                features = extract_fft_features(chunk_signal)
                
                if features is not None:
                    raw_features.append(features)
                    raw_labels.append(category)
                    category_features.append(features)
                    print(f"    ✓ Chunk {i}: {len(features)} features")
                else:
                    print(f"    ❌ Failed to extract features from chunk {i}")
            
            if category_features:
                print(f"  📊 {category} statistics:")
                print(f"    Chunks processed: {len(category_features)}")
                print(f"    Feature dimensions: {len(category_features[0])}")
                if len(category_features) > 1:
                    feature_matrix = np.array(category_features)
                    print(f"    Feature means: {np.mean(feature_matrix, axis=0)[:5]}... (first 5)")
                    print(f"    Feature stds: {np.std(feature_matrix, axis=0)[:5]}... (first 5)")
        else:
            print(f"  ❌ File not found: {file_path}")
    
    print(f"\n📋 RAW DATA SUMMARY")
    print("=" * 30)
    print(f"Total chunks processed: {len(raw_features)}")
    print(f"Total labels: {len(raw_labels)}")
    
    if raw_features:
        feature_dims = [len(f) for f in raw_features]
        print(f"Feature dimensions: {set(feature_dims)}")
        print(f"All same dimension: {len(set(feature_dims)) == 1}")
        
        if len(set(feature_dims)) == 1:
            # All features have same dimension, test classification setup
            raw_features_array = np.array(raw_features)
            raw_labels_array = np.array(raw_labels)
            
            print(f"\n🎯 Classification task setup:")
            
            # Human detection
            human_mask = (raw_labels_array == 'human') | (raw_labels_array == 'human_nothing')
            if np.sum(human_mask) > 0:
                human_features = raw_features_array[human_mask]
                human_labels = (raw_labels_array[human_mask] == 'human').astype(int)
                print(f"  Human detection: {len(human_features)} samples")
                print(f"    Class distribution: {np.bincount(human_labels)}")
            
            # Car detection  
            car_mask = (raw_labels_array == 'car') | (raw_labels_array == 'car_nothing')
            if np.sum(car_mask) > 0:
                car_features = raw_features_array[car_mask]
                car_labels = (raw_labels_array[car_mask] == 'car').astype(int)
                print(f"  Car detection: {len(car_features)} samples")
                print(f"    Class distribution: {np.bincount(car_labels)}")
        
        print(f"\n✅ Raw FFT processing test successful!")
        return True
    else:
        print(f"\n❌ No features extracted successfully!")
        return False

if __name__ == "__main__":
    test_raw_fft_processing() 