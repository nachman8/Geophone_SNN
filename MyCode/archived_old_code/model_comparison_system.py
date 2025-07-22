#!/usr/bin/env python3
"""
Comprehensive Model Training and Comparison System
==================================================

Trains and compares different machine learning models on:
1. Chunk data: Resonator-processed spike features (32D discriminative features)
2. Raw data: FFT-processed signal features (frequency domain features)

Models tested:
- Weighted Ensemble
- SVM
- XGBoost
- Random Forest
- LightGBM
- KNN

Classification tasks:
- Human vs Human_nothing (footprint detection)
- Car vs Car_nothing (car detection)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import spectrogram, welch
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
RAW_DATA_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/data"
RESULTS_DIR = "model_comparison_results"
CHUNK_DURATION = 30  # seconds
SAMPLING_FREQ = 1000  # Hz

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("🤖 COMPREHENSIVE MODEL TRAINING AND COMPARISON SYSTEM")
print("=" * 70)
print("📊 Comparing Chunk (Resonator) vs Raw (FFT) Data Performance")
print("🎯 Tasks: Human Detection & Car Detection")
print("🧠 Models: Ensemble, SVM, XGBoost, RF, LightGBM, KNN")
print("=" * 70)

# ========================================================================
# CHUNK DATA LOADING AND FEATURE EXTRACTION
# ========================================================================

def load_chunk_data_features():
    """Load and extract features from chunk (resonator) data"""
    print("\n📁 Loading Chunk Data (Resonator Features)...")
    
    chunk_features = []
    chunk_labels = []
    
    categories = ['human', 'human_nothing', 'car', 'car_nothing']
    
    for category in categories:
        category_dir = Path(CHUNKED_OUTPUT_DIR) / category
        print(f"Processing {category}...")
        
        # Get all chunk directories
        chunk_dirs = [d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith('chunk_')]
        
        for chunk_dir in chunk_dirs[:10]:  # Use more chunks for better training
            pkl_file = chunk_dir / f"{chunk_dir.name}_data.pkl"
            
            if pkl_file.exists():
                try:
                    # Load the chunk data
                    with open(pkl_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Extract features from the resonator output
                    features = extract_resonator_features(chunk_data)
                    
                    if features is not None:
                        chunk_features.append(features)
                        chunk_labels.append(category)
                        print(f"  Loaded {chunk_dir.name}: {len(features)} features")
                        
                except Exception as e:
                    print(f"  Error loading {pkl_file}: {e}")
    
    return chunk_features, chunk_labels

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

# ========================================================================
# RAW DATA LOADING AND FFT FEATURE EXTRACTION
# ========================================================================

def load_raw_data_features():
    """Load raw data and extract FFT features in 30-second windows"""
    print("\n📊 Loading Raw Data (FFT Features)...")
    
    raw_features = []
    raw_labels = []
    
    files = {
        'human': Path(RAW_DATA_DIR) / 'human.csv',
        'human_nothing': Path(RAW_DATA_DIR) / 'human_nothing.csv',
        'car': Path(RAW_DATA_DIR) / 'car.csv',
        'car_nothing': Path(RAW_DATA_DIR) / 'car_nothing.csv'
    }
    
    for category, file_path in files.items():
        print(f"Processing {category}...")
        
        if file_path.exists():
            # Load the CSV file
            df = pd.read_csv(file_path)
            signal = df['amplitude'].values
            
            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Split into 30-second chunks
            chunk_size = CHUNK_DURATION * SAMPLING_FREQ  # 30 seconds * 1000 Hz
            num_chunks = len(signal) // chunk_size
            
            for i in range(min(num_chunks, 20)):  # Use more chunks for better training
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk_signal = signal[start_idx:end_idx]
                
                # Extract FFT features
                features = extract_fft_features(chunk_signal)
                
                if features is not None:
                    raw_features.append(features)
                    raw_labels.append(category)
                    
            print(f"  Created {min(num_chunks, 10)} chunks from {category}")
    
    return raw_features, raw_labels

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

# ========================================================================
# MODEL DEFINITIONS
# ========================================================================

def create_models():
    """Create all models for comparison"""
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }
    
    # Weighted Ensemble
    ensemble = VotingClassifier([
        ('svm', SVC(probability=True, random_state=42)),
        ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss')),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ], voting='soft')
    
    models['Weighted Ensemble'] = ensemble
    
    return models

# ========================================================================
# TRAINING AND EVALUATION
# ========================================================================

def train_and_evaluate_models(X, y, task_name, data_type):
    """Train and evaluate all models"""
    print(f"\n🚀 Training Models for {task_name} - {data_type} Data")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create models
    models = create_models()
    
    results = {}
    
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            train_time = time.time() - start_time
            
            results[model_name] = {
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_time': train_time,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"    ✓ Accuracy: {test_accuracy:.3f} (±{cv_scores.std():.3f})")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[model_name] = {
                'test_accuracy': 0,
                'cv_mean': 0,
                'cv_std': 0,
                'train_time': 0,
                'error': str(e)
            }
    
    return results

# ========================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ========================================================================

def create_comparison_table(all_results):
    """Create comprehensive comparison table"""
    print("\n📊 COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 80)
    
    # Prepare data for DataFrame
    table_data = []
    
    for (task, data_type), results in all_results.items():
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                table_data.append({
                    'Task': task,
                    'Data Type': data_type,
                    'Model': model_name,
                    'Test Accuracy': f"{metrics['test_accuracy']:.3f}",
                    'CV Mean': f"{metrics['cv_mean']:.3f}",
                    'CV Std': f"{metrics['cv_std']:.3f}",
                    'Train Time (s)': f"{metrics['train_time']:.2f}"
                })
    
    df_results = pd.DataFrame(table_data)
    
    # Save results
    df_results.to_csv(f"{RESULTS_DIR}/model_comparison_results.csv", index=False)
    
    # Display formatted table
    print(df_results.to_string(index=False))
    
    # Create pivot tables for better visualization
    pivot_accuracy = df_results.pivot_table(
        values='Test Accuracy', 
        index=['Task', 'Model'], 
        columns='Data Type',
        aggfunc='first'
    )
    
    print(f"\n📈 ACCURACY COMPARISON BY DATA TYPE")
    print("=" * 50)
    print(pivot_accuracy.to_string())
    
    return df_results

def plot_results(df_results):
    """Create visualization plots"""
    print("\n📊 Creating visualization plots...")
    
    # Convert accuracy columns to numeric
    df_results['Test Accuracy'] = pd.to_numeric(df_results['Test Accuracy'])
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy comparison by model and data type
    ax1 = axes[0, 0]
    df_pivot = df_results.pivot_table(
        values='Test Accuracy',
        index='Model',
        columns='Data Type',
        aggfunc='mean'
    )
    df_pivot.plot(kind='bar', ax=ax1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title('Model Accuracy by Data Type')
    ax1.set_ylabel('Test Accuracy')
    ax1.legend(title='Data Type')
    
    # Plot 2: Task-specific performance
    ax2 = axes[0, 1]
    task_perf = df_results.groupby(['Task', 'Data Type'])['Test Accuracy'].mean().unstack()
    task_perf.plot(kind='bar', ax=ax2)
    ax2.set_title('Performance by Task')
    ax2.set_ylabel('Average Test Accuracy')
    ax2.set_xlabel('Task')
    
    # Plot 3: Training time comparison
    ax3 = axes[1, 0]
    df_results['Train Time (s)'] = pd.to_numeric(df_results['Train Time (s)'])
    time_pivot = df_results.pivot_table(
        values='Train Time (s)',
        index='Model',
        columns='Data Type',
        aggfunc='mean'
    )
    time_pivot.plot(kind='bar', ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_title('Training Time Comparison')
    ax3.set_ylabel('Training Time (seconds)')
    
    # Plot 4: Best model per task
    ax4 = axes[1, 1]
    best_models = df_results.loc[df_results.groupby(['Task', 'Data Type'])['Test Accuracy'].idxmax()]
    best_count = best_models['Model'].value_counts()
    best_count.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
    ax4.set_title('Best Model Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    """Main execution function"""
    print("🔍 Step 1: Examining chunk data structure...")
    
    # First, let's examine a sample chunk file
    sample_chunk = Path(CHUNKED_OUTPUT_DIR) / "human" / "chunk_0" / "chunk_0_data.pkl"
    if sample_chunk.exists():
        with open(sample_chunk, 'rb') as f:
            sample_data = pickle.load(f)
        print(f"Sample chunk data type: {type(sample_data)}")
        if isinstance(sample_data, dict):
            print(f"Keys: {list(sample_data.keys())}")
            for key, value in sample_data.items():
                print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
    
    all_results = {}
    
    try:
        # Load chunk data
        chunk_X, chunk_y = load_chunk_data_features()
        print(f"Loaded chunk data: {len(chunk_X)} samples")
        
        # Load raw data
        raw_X, raw_y = load_raw_data_features()
        print(f"Loaded raw data: {len(raw_X)} samples")
        
        # Prepare classification tasks
        tasks = {}
        
        # Separate chunk data by task (since they have different feature dimensions)
        chunk_human_X, chunk_human_y = [], []
        chunk_car_X, chunk_car_y = [], []
        
        for i, label in enumerate(chunk_y):
            if label in ['human', 'human_nothing']:
                chunk_human_X.append(chunk_X[i])
                chunk_human_y.append(label)
            elif label in ['car', 'car_nothing']:
                chunk_car_X.append(chunk_X[i])
                chunk_car_y.append(label)
        
        # Human detection task (chunk data)
        if len(chunk_human_X) > 0:
            chunk_human_X = np.array(chunk_human_X)
            chunk_human_y = np.array(chunk_human_y)
            tasks[('Human Detection', 'Chunk')] = (
                chunk_human_X, 
                (chunk_human_y == 'human').astype(int)
            )
        
        # Car detection task (chunk data)
        if len(chunk_car_X) > 0:
            chunk_car_X = np.array(chunk_car_X)
            chunk_car_y = np.array(chunk_car_y)
            tasks[('Car Detection', 'Chunk')] = (
                chunk_car_X, 
                (chunk_car_y == 'car').astype(int)
            )
        
        # Raw data tasks (all have same feature dimensions)
        if len(raw_X) > 0:
            raw_X = np.array(raw_X)
            raw_y = np.array(raw_y)
            
            # Human detection task (raw data)
            human_mask_raw = (raw_y == 'human') | (raw_y == 'human_nothing')
            if np.sum(human_mask_raw) > 0:
                tasks[('Human Detection', 'Raw FFT')] = (
                    raw_X[human_mask_raw], 
                    (raw_y[human_mask_raw] == 'human').astype(int)
                )
            
            # Car detection task (raw data)
            car_mask_raw = (raw_y == 'car') | (raw_y == 'car_nothing')
            if np.sum(car_mask_raw) > 0:
                tasks[('Car Detection', 'Raw FFT')] = (
                    raw_X[car_mask_raw], 
                    (raw_y[car_mask_raw] == 'car').astype(int)
                )
        
        # Train and evaluate models for each task
        for (task_name, data_type), (X, y) in tasks.items():
            results = train_and_evaluate_models(X, y, task_name, data_type)
            all_results[(task_name, data_type)] = results
        
        # Create comparison table and plots
        if all_results:
            df_results = create_comparison_table(all_results)
            plot_results(df_results)
            
            print(f"\n✅ Results saved to {RESULTS_DIR}/")
            print("📊 Analysis complete!")
        else:
            print("❌ No valid data found for training!")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 