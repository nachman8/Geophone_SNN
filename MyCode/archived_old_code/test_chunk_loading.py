#!/usr/bin/env python3
"""
Test Chunk Data Loading
=======================

This script tests the chunk data loading and feature extraction functions
to ensure they work correctly before running the full model comparison.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

def extract_resonator_features(chunk_data):
    """Extract discriminative features from resonator chunk data"""
    try:
        if isinstance(chunk_data, dict):
            features = []
            
            # Extract statistical features from max_spikes_spectrogram
            if 'max_spikes_spectrogram' in chunk_data:
                max_spikes_spec = chunk_data['max_spikes_spectrogram']
                if isinstance(max_spikes_spec, np.ndarray):
                    # Extract statistical features for each resonator
                    for resonator_idx in range(max_spikes_spec.shape[0]):
                        resonator_data = max_spikes_spec[resonator_idx, :]
                        features.extend([
                            np.mean(resonator_data),      # Mean activity
                            np.std(resonator_data),       # Variability
                            np.max(resonator_data),       # Peak activity
                            np.sum(resonator_data > 0),   # Active time bins
                            np.percentile(resonator_data, 75),  # 75th percentile
                            np.percentile(resonator_data, 25),  # 25th percentile
                        ])
            
            # Extract statistical features from spikes_bands_spectrogram
            if 'spikes_bands_spectrogram' in chunk_data:
                bands_spec = chunk_data['spikes_bands_spectrogram']
                if isinstance(bands_spec, np.ndarray):
                    # Extract statistical features for each frequency band
                    for band_idx in range(bands_spec.shape[0]):
                        band_data = bands_spec[band_idx, :]
                        features.extend([
                            np.mean(band_data),           # Mean band power
                            np.std(band_data),            # Band variability
                            np.max(band_data),            # Peak band power
                            np.sum(band_data > 0),        # Active time bins
                        ])
            
            # Add signal statistics
            if 'signal' in chunk_data:
                signal = chunk_data['signal']
                if isinstance(signal, np.ndarray) and len(signal) > 0:
                    features.extend([
                        np.mean(signal),
                        np.std(signal),
                        np.max(signal),
                        np.min(signal),
                        np.sqrt(np.mean(signal**2)),  # RMS
                        np.percentile(signal, 75),    # 75th percentile
                        np.percentile(signal, 25),    # 25th percentile
                    ])
            
            return np.array(features) if features else None
            
        else:
            print(f"Unexpected chunk data structure: {type(chunk_data)}")
            return None
            
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def test_chunk_loading():
    """Test chunk data loading and feature extraction"""
    print("🧪 TESTING CHUNK DATA LOADING")
    print("=" * 50)
    
    chunk_features = []
    chunk_labels = []
    feature_info = {}
    
    categories = ['human', 'human_nothing', 'car', 'car_nothing']
    
    for category in categories:
        print(f"\n📂 Testing {category}...")
        category_dir = Path(CHUNKED_OUTPUT_DIR) / category
        
        # Get chunk directories
        chunk_dirs = [d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith('chunk_')]
        chunk_dirs = sorted(chunk_dirs, key=lambda x: int(x.name.split('_')[1]))
        
        category_features = []
        
        for chunk_dir in chunk_dirs[:3]:  # Test first 3 chunks
            pkl_file = chunk_dir / f"{chunk_dir.name}_data.pkl"
            
            if pkl_file.exists():
                try:
                    # Load the chunk data
                    with open(pkl_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Extract features
                    features = extract_resonator_features(chunk_data)
                    
                    if features is not None:
                        chunk_features.append(features)
                        chunk_labels.append(category)
                        category_features.append(features)
                        
                        print(f"  ✓ {chunk_dir.name}: {len(features)} features")
                        
                        # Analyze feature composition
                        if category not in feature_info:
                            feature_info[category] = {
                                'max_spikes_len': len(chunk_data.get('max_spikes_spectrogram', [])),
                                'bands_len': len(chunk_data.get('spikes_bands_spectrogram', [])),
                                'signal_stats': 5,  # mean, std, max, min, rms
                                'total_features': len(features)
                            }
                        
                except Exception as e:
                    print(f"  ❌ Error loading {pkl_file}: {e}")
        
        if category_features:
            print(f"  📊 {category} statistics:")
            print(f"    Chunks loaded: {len(category_features)}")
            print(f"    Feature dimensions: {len(category_features[0]) if category_features else 0}")
            if len(category_features) > 1:
                feature_matrix = np.array(category_features)
                print(f"    Feature means: {np.mean(feature_matrix, axis=0)[:5]}... (first 5)")
                print(f"    Feature stds: {np.std(feature_matrix, axis=0)[:5]}... (first 5)")
    
    print(f"\n📋 SUMMARY")
    print("=" * 30)
    print(f"Total chunks loaded: {len(chunk_features)}")
    print(f"Total labels: {len(chunk_labels)}")
    
    if chunk_features:
        print(f"Feature dimensions: {len(chunk_features[0])}")
        
        print(f"\n📊 Feature composition by category:")
        for category, info in feature_info.items():
            print(f"  {category}:")
            print(f"    Max spikes features: {info['max_spikes_len']}")
            print(f"    Band features: {info['bands_len']}")
            print(f"    Signal statistics: {info['signal_stats']}")
            print(f"    Total features: {info['total_features']}")
        
        # Test creating train/test split structure
        chunk_features_array = np.array(chunk_features)
        chunk_labels_array = np.array(chunk_labels)
        
        print(f"\n🎯 Classification task setup:")
        
        # Human detection
        human_mask = (chunk_labels_array == 'human') | (chunk_labels_array == 'human_nothing')
        if np.sum(human_mask) > 0:
            human_features = chunk_features_array[human_mask]
            human_labels = (chunk_labels_array[human_mask] == 'human').astype(int)
            print(f"  Human detection: {len(human_features)} samples")
            print(f"    Class distribution: {np.bincount(human_labels)}")
        
        # Car detection  
        car_mask = (chunk_labels_array == 'car') | (chunk_labels_array == 'car_nothing')
        if np.sum(car_mask) > 0:
            car_features = chunk_features_array[car_mask]
            car_labels = (chunk_labels_array[car_mask] == 'car').astype(int)
            print(f"  Car detection: {len(car_features)} samples")
            print(f"    Class distribution: {np.bincount(car_labels)}")
        
        print(f"\n✅ Chunk loading test successful!")
        return True
    else:
        print(f"\n❌ No chunks loaded successfully!")
        return False

if __name__ == "__main__":
    test_chunk_loading() 