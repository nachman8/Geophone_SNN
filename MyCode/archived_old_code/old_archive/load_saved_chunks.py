#!/usr/bin/env python3
"""
Direct SNN Training from Saved Chunks
Bypasses the chunking process and directly loads existing chunk files
"""

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import sys

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sklearn.model_selection import train_test_split

def load_chunks_directly(chunks_base_dir):
    """
    Directly load existing chunk files from saved directories
    """
    print(f"🔄 Loading saved chunks from {chunks_base_dir}")
    
    chunk_data = {}
    
    # Look for car and car_nothing directories
    for signal_type in ['car', 'car_nothing', 'human', 'human_nothing']:
        chunk_dir = os.path.join(chunks_base_dir, signal_type)
        index_file = os.path.join(chunk_dir, 'chunk_index.pkl')
        
        if os.path.exists(index_file):
            print(f"📁 Loading {signal_type} chunks...")
            
            # Load chunk index
            with open(index_file, 'rb') as f:
                chunk_index = pickle.load(f)
            
            # Load all chunk data files
            chunks = []
            for chunk_file in chunk_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk = pickle.load(f)
                    chunks.append(chunk)
                else:
                    print(f"⚠️  Chunk file not found: {chunk_file}")
            
            chunk_data[signal_type] = {
                'index': chunk_index,
                'chunks': chunks
            }
            
            print(f"   ✅ Loaded {len(chunks)} chunks for {signal_type}")
        else:
            print(f"   ❌ No chunk index found for {signal_type}")
    
    return chunk_data

def extract_segments_from_loaded_chunks(loaded_chunks, signal_type='car'):
    """
    Extract training segments from loaded chunks using FIXED threshold detection
    """
    segments = []
    labels = []
    
    print(f"\n📊 EXTRACTING SEGMENTS FROM LOADED {signal_type.upper()} CHUNKS")
    
    if signal_type.endswith('_nothing'):
        file_type = 'nothing'
        base_signal_type = signal_type.replace('_nothing', '')
        print(f"Processing {len(loaded_chunks['chunks'])} chunks from {signal_type}.csv (type: nothing)")
    else:
        file_type = 'signal'
        base_signal_type = signal_type
        print(f"Processing {len(loaded_chunks['chunks'])} chunks from {signal_type}.csv (type: signal)")
        
    chunks = loaded_chunks['chunks']
    for chunk_idx, chunk in enumerate(chunks):
        if 'spikes_bands_spectrogram' not in chunk:
            print(f"Warning: Invalid chunk {chunk_idx}")
            continue
            
        chunk_spikes_bands_spectrogram = chunk['spikes_bands_spectrogram']
        chunk_duration = chunk['duration']
        
        # FIXED threshold detection with proper parameters
        chunk_segments, chunk_labels, chunk_confidence = fixed_signal_detection(
            chunk_spikes_bands_spectrogram, 
            chunk_duration, 
            base_signal_type, 
            file_type
        )
        
        segments.extend(chunk_segments)
        labels.extend(chunk_labels)
        
        signal_count = np.sum(np.array(chunk_labels) == 1)
        nothing_count = np.sum(np.array(chunk_labels) == 0)
        
        print(f"Chunk {chunk_idx}: {len(chunk_segments)} segments, "
              f"{signal_count} with signal, {nothing_count} without")
    
    return np.array(segments), np.array(labels)

def fixed_signal_detection(spikes_bands_spectrogram, duration, signal_type='car', file_type='signal'):
    """
    IMPROVED signal detection that properly distinguishes signal vs nothing files
    """
    n_bands, n_time_bins = spikes_bands_spectrogram.shape
    
    # Adaptive parameters
    if signal_type == 'car':
        segment_duration = 15
        important_bands = [1, 2, 3, 4]  # CAR bands: 30-50 Hz
    else:
        segment_duration = 7
        important_bands = [5, 6, 7]  # HUMAN bands: 60-85 Hz
    
    samples_per_segment = int(segment_duration * 100)
    
    segments = []
    segment_labels = []
    segment_confidence = []
    
    # Handle small data
    if n_time_bins < samples_per_segment:
        segment_data = spikes_bands_spectrogram
        segment_features = []
        
        for band_idx in range(n_bands):
            band_data = segment_data[band_idx]
            features = [
                np.mean(band_data) if len(band_data) > 0 else 0,
                np.max(band_data) if len(band_data) > 0 else 0,
                np.std(band_data) if len(band_data) > 0 else 0,
                np.sum(band_data > np.mean(band_data) + np.std(band_data)) if len(band_data) > 0 else 0,
                np.sum(band_data > 0) / len(band_data) if len(band_data) > 0 else 0,
                np.percentile(band_data, 90) if len(band_data) > 0 else 0,
                np.sum(np.diff(band_data) > 0) / len(band_data) if len(band_data) > 1 else 0,
            ]
            segment_features.extend(features)
        
        segments.append(segment_features)
        segment_labels.append(0)
        segment_confidence.append(0.0)
        
        return np.array(segments), np.array(segment_labels), np.array(segment_confidence)
    
    # Process segments
    for start_idx in range(0, n_time_bins - samples_per_segment, samples_per_segment // 2):
        end_idx = start_idx + samples_per_segment
        segment_data = spikes_bands_spectrogram[:, start_idx:end_idx]
        
        # Calculate comprehensive activity metrics
        important_activity = np.sum(segment_data[important_bands])
        total_activity = np.sum(segment_data)
        mean_activity = np.mean(segment_data)
        std_activity = np.std(segment_data)
        max_activity = np.max(segment_data)
        
        # Activity ratio
        activity_ratio = important_activity / (total_activity + 1e-10)
        
        # Signal strength
        signal_strength = mean_activity + std_activity
        overall_baseline = np.mean(spikes_bands_spectrogram)
        
        # FIXED DETECTION LOGIC based on file type
        if file_type == 'nothing':
            # For nothing files: be VERY conservative, require STRONG evidence for signal
            if signal_type == 'car':
                # Car nothing should have very low organized activity
                has_signal = (
                    (activity_ratio > 0.35) and  # INCREASED from 0.25 - Important bands must strongly dominate
                    (signal_strength > 2.5 * overall_baseline) and  # INCREASED from 1.5 - Must be well above baseline
                    (max_activity > 3.0 * mean_activity) and  # INCREASED from 2.0 - Clear peaks required
                    (mean_activity > 0.1 * np.max(spikes_bands_spectrogram))  # NEW - minimum activity level
                )
            else:
                # Human nothing should have very low burst activity
                important_bands_data = segment_data[important_bands]
                if important_bands_data.size > 0:
                    activity_pattern = np.mean(important_bands_data, axis=0)
                    if len(activity_pattern) > 10:
                        threshold = np.mean(activity_pattern) + 3.0 * np.std(activity_pattern)  # INCREASED from 2.0
                        bursts = activity_pattern > threshold
                        n_bursts = np.sum(np.diff(np.concatenate([[False], bursts, [False]])) == 1)
                        burst_score = n_bursts / len(activity_pattern) * 100
                    else:
                        burst_score = 0
                else:
                    burst_score = 0
                
                has_signal = (
                    (activity_ratio > 0.40) and  # INCREASED from 0.30 - Important bands must strongly dominate
                    (signal_strength > 3.0 * overall_baseline) and  # INCREASED from 1.8 - Well above baseline
                    (burst_score > 2.0) and  # INCREASED from 1.0 - Clear burst pattern required
                    (mean_activity > 0.15 * np.max(spikes_bands_spectrogram))  # NEW - minimum activity level
                )
        else:
            # For signal files: be more sensitive, expect organized activity
            if signal_type == 'car':
                # Car signal: look for consistent mid-frequency activity
                important_bands_data = segment_data[important_bands]
                if important_bands_data.size > 0:
                    important_strength = np.mean(important_bands_data)
                    overall_strength = np.mean(segment_data)
                    car_score = important_strength / (overall_strength + 1e-10)
                else:
                    car_score = 0
                
                has_signal = (
                    (activity_ratio > 0.15 and signal_strength > 0.8 * overall_baseline) or  # Require BOTH conditions
                    (car_score > 1.15 and signal_strength > 1.0 * overall_baseline) or  # Higher car score requirement
                    (activity_ratio > 0.25)  # Very strong activity ratio requirement
                )
            else:
                # Human signal: look for burst patterns
                important_bands_data = segment_data[important_bands]
                if important_bands_data.size > 0:
                    activity_pattern = np.mean(important_bands_data, axis=0)
                    if len(activity_pattern) > 10:
                        threshold = np.mean(activity_pattern) + 0.8 * np.std(activity_pattern)  # DECREASED from 1.0
                        bursts = activity_pattern > threshold
                        n_bursts = np.sum(np.diff(np.concatenate([[False], bursts, [False]])) == 1)
                        burst_score = n_bursts / len(activity_pattern) * 100
                    else:
                        burst_score = 0
                    
                    human_strength = np.mean(important_bands_data)
                    overall_strength = np.mean(segment_data)
                    human_score = human_strength / (overall_strength + 1e-10)
                else:
                    burst_score = 0
                    human_score = 0
                
                has_signal = (
                    (activity_ratio > 0.18 and signal_strength > 0.9 * overall_baseline) or  # Require BOTH conditions
                    (burst_score > 0.25 and human_score > 1.2) or  # Higher burst + human score requirements
                    (activity_ratio > 0.30)  # Very strong activity ratio requirement
                )
        
        # Create feature vector
        segment_features = []
        for band_idx in range(n_bands):
            band_data = segment_data[band_idx]
            if len(band_data) > 0:
                features = [
                    np.mean(band_data),
                    np.max(band_data),
                    np.std(band_data),
                    np.sum(band_data > np.mean(band_data) + np.std(band_data)),
                    np.sum(band_data > 0) / len(band_data),
                    np.percentile(band_data, 90),
                    np.sum(np.diff(band_data) > 0) / len(band_data) if len(band_data) > 1 else 0,
                ]
            else:
                features = [0, 0, 0, 0, 0, 0, 0]
            segment_features.extend(features)
        
        segments.append(segment_features)
        segment_labels.append(1 if has_signal else 0)
        segment_confidence.append(activity_ratio)
    
    return np.array(segments), np.array(segment_labels), np.array(segment_confidence)



def prepare_binary_training_data(chunk_data, signal_type_base):
    """
    Prepare training data for binary classification (signal vs nothing)
    """
    print(f"\n🎯 PREPARING {signal_type_base.upper()} BINARY TRAINING DATA")
    
    signal_file = signal_type_base  # e.g., 'car'
    nothing_file = f"{signal_type_base}_nothing"  # e.g., 'car_nothing'
    
    all_segments = []
    all_labels = []
    
    # Process signal file
    if signal_file in chunk_data:
        print(f"📁 Processing {signal_file} chunks...")
        segments, labels = extract_segments_from_loaded_chunks(chunk_data[signal_file], signal_file)
        
        # Keep segments with signal activity for signal class
        signal_indices = labels == 1
        signal_segments = segments[signal_indices]
        print(f"   Found {len(signal_segments)} {signal_file} signal segments")
        
        for segment in signal_segments:
            all_segments.append(segment)
            all_labels.append(0)  # 0 = signal for SNN
    
    # Process nothing file
    if nothing_file in chunk_data:
        print(f"📁 Processing {nothing_file} chunks...")
        segments, labels = extract_segments_from_loaded_chunks(chunk_data[nothing_file], nothing_file)
        
        # Keep segments WITHOUT signal activity for nothing class
        nothing_indices = labels == 0
        nothing_segments = segments[nothing_indices]
        print(f"   Found {len(nothing_segments)} {nothing_file} nothing segments")
        
        for segment in nothing_segments:
            all_segments.append(segment)
            all_labels.append(1)  # 1 = nothing for SNN
    
    final_segments = np.array(all_segments)
    final_labels = np.array(all_labels)
    
    print(f"\n📊 FINAL {signal_type_base.upper()} DATASET:")
    print(f"Total segments: {len(final_segments)}")
    print(f"Signal segments: {np.sum(final_labels == 0)}")
    print(f"Nothing segments: {np.sum(final_labels == 1)}")
    
    return final_segments, final_labels

def train_from_saved_chunks(chunks_base_dir):
    """
    Main function to train SNN directly from saved chunks
    """
    print("🚀 DIRECT SNN TRAINING FROM SAVED CHUNKS")
    print("=" * 60)
    
    # Load saved chunks
    chunk_data = load_chunks_directly(chunks_base_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return None
    
    results = {}
    
    # Train Car Classification
    if 'car' in chunk_data or 'car_nothing' in chunk_data:
        print(f"\n🚗 TRAINING CAR vs CAR_NOTHING CLASSIFICATION")
        car_segments, car_labels = prepare_binary_training_data(chunk_data, 'car')
        
        if len(car_segments) > 0 and len(np.unique(car_labels)) >= 2:
            car_results = train_binary_snn(car_segments, car_labels, 'car')
            results['car'] = car_results
        else:
            print(f"❌ Insufficient car data for training")
            results['car'] = None
    
    # Train Human Classification
    if 'human' in chunk_data or 'human_nothing' in chunk_data:
        print(f"\n👤 TRAINING HUMAN vs HUMAN_NOTHING CLASSIFICATION")
        human_segments, human_labels = prepare_binary_training_data(chunk_data, 'human')
        
        if len(human_segments) > 0 and len(np.unique(human_labels)) >= 2:
            human_results = train_binary_snn(human_segments, human_labels, 'human')
            results['human'] = human_results
        else:
            print(f"❌ Insufficient human data for training")
            results['human'] = None
    
    return results

def train_binary_snn(segments, labels, signal_type):
    """
    Train binary SNN classification
    """
    print(f"\n🧠 Training {signal_type.upper()} SNN...")
    
    # Import SNN
    from snn_classification import GeophoneSNN
    
    # Create SNN
    snn = GeophoneSNN(n_hidden=40, learning_rate=0.015)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            segments, labels, test_size=0.25, random_state=42, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            segments, labels, test_size=0.25, random_state=42
        )
    
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Train
    training_history = snn.train(X_train, y_train, n_epochs=60, spike_duration=200)
    
    # Evaluate
    accuracy, report, cm = snn.evaluate(X_test, y_test)
    
    # Save model
    model_path = f"direct_{signal_type}_snn_model.pkl"
    snn.save_model(model_path)
    
    print(f"✅ {signal_type.upper()} SNN Complete!")
    print(f"📊 Accuracy: {accuracy:.1%}")
    print(f"💾 Model: {model_path}")
    
    return {
        'snn': snn,
        'accuracy': accuracy,
        'training_history': training_history,
        'model_path': model_path
    }

if __name__ == "__main__":
    # Set the chunks directory
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    print("🎯 DIRECT TRAINING FROM SAVED CHUNKS")
    print("This script bypasses chunk creation and directly uses saved chunk files")
    print()
    
    # Train from saved chunks
    results = train_from_saved_chunks(chunks_dir)
    
    if results:
        print(f"\n🎉 DIRECT TRAINING RESULTS:")
        
        if results.get('car'):
            car_res = results['car']
            print(f"\n🚗 CAR CLASSIFICATION:")
            print(f"   ✅ Accuracy: {car_res['accuracy']:.1%}")
            print(f"   💾 Model: {car_res['model_path']}")
        
        if results.get('human'):
            human_res = results['human']
            print(f"\n👤 HUMAN CLASSIFICATION:")
            print(f"   ✅ Accuracy: {human_res['accuracy']:.1%}")
            print(f"   💾 Model: {human_res['model_path']}")
    else:
        print("❌ Direct training failed") 