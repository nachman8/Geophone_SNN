#!/usr/bin/env python3
"""
Comprehensive Geophone Classification System
Following EEG Study Methodology: FFT vs SCTN Feature Comparison
Based on "SCTN-based EEG classification" approach
"""

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import sctnN components
from sctnN.spiking_network import SpikingNetwork
from sctnN.layers import SCTNLayer
from sctnN.spiking_neuron import create_SCTN, BINARY, IDENTITY
from sctnN.resonator_functions import get_closest_resonator

class FFTFeatureExtractor:
    """
    FFT-based feature extraction following EEG study methodology
    """
    
    def __init__(self, frame_durations=[5, 10, 15, 20]):
        self.frame_durations = frame_durations  # in ms, like EEG study
        self.frequency_bands = {
            'LOW_FREQ': (20, 30),      # Environmental noise
            'CAR_APPROACH': (30, 34),  # Vehicle approach
            'CAR_PEAK': (34, 40),      # Main vehicle signature  
            'CAR_TAIL': (40, 48),      # Vehicle departure
            'MID_GAP': (48, 60),       # Transition band
            'HUMAN_PEAK': (60, 70),    # Primary footstep frequency
            'HUMAN_TAIL': (70, 85),    # Secondary footstep harmonics
            'HIGH_FREQ': (85, 100)     # High frequency noise
        }
        
        print(f"🔊 FFTFeatureExtractor initialized")
        print(f"   📊 Frame durations: {frame_durations} ms")
        print(f"   🎯 {len(self.frequency_bands)} frequency bands")
    
    def extract_features(self, spikes_bands_spectrogram, duration, frame_duration_ms=10):
        """
        Extract FFT-based features from spikegram data
        Following EEG study approach with different frame durations
        """
        features = []
        
        # Convert spikegram to time-domain signal for FFT analysis
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        # For each frequency band, extract FFT features
        for band_idx, (band_name, (fmin, fmax)) in enumerate(self.frequency_bands.items()):
            if band_idx < n_bands:
                band_signal = spikes_bands_spectrogram[band_idx, :]
                
                # Frame-based FFT analysis (like EEG study)
                frame_features = self._extract_frame_fft_features(
                    band_signal, frame_duration_ms, duration
                )
                features.extend(frame_features)
        
        # Global spectral features
        global_features = self._extract_global_spectral_features(spikes_bands_spectrogram)
        features.extend(global_features)
        
        return np.array(features)
    
    def _extract_frame_fft_features(self, signal, frame_duration_ms, total_duration):
        """Extract FFT features from signal frames"""
        features = []
        
        if len(signal) == 0:
            return [0] * 8  # Return zeros if no signal
        
        # Calculate frame size
        sampling_rate = len(signal) / total_duration  # samples per second
        frame_size = int(frame_duration_ms * sampling_rate / 1000)
        
        if frame_size <= 1:
            frame_size = min(len(signal), 10)
        
        # Extract frames and compute FFT
        n_frames = max(1, len(signal) // frame_size)
        frame_powers = []
        
        for i in range(n_frames):
            start_idx = i * frame_size
            end_idx = min(start_idx + frame_size, len(signal))
            frame = signal[start_idx:end_idx]
            
            if len(frame) > 1:
                # Compute FFT
                fft_vals = fft(frame)
                power_spectrum = np.abs(fft_vals)**2
                frame_powers.append(np.mean(power_spectrum))
            else:
                frame_powers.append(0)
        
        if frame_powers:
            # Statistical features from frame powers
            features.extend([
                np.mean(frame_powers),      # Mean power
                np.std(frame_powers),       # Power variability
                np.max(frame_powers),       # Peak power
                np.min(frame_powers),       # Minimum power
                np.median(frame_powers),    # Median power
                np.percentile(frame_powers, 75) - np.percentile(frame_powers, 25),  # IQR
                np.sum(frame_powers),       # Total energy
                len([p for p in frame_powers if p > np.mean(frame_powers)])  # Active frames
            ])
        else:
            features.extend([0] * 8)
        
        return features
    
    def _extract_global_spectral_features(self, spikes_bands_spectrogram):
        """Extract global spectral characteristics"""
        features = []
        
        # Overall signal characteristics
        total_energy = np.sum(spikes_bands_spectrogram)
        features.append(total_energy)
        
        # Band energy distribution
        for band_idx in range(spikes_bands_spectrogram.shape[0]):
            band_energy = np.sum(spikes_bands_spectrogram[band_idx, :])
            band_ratio = band_energy / (total_energy + 1e-10)
            features.append(band_ratio)
        
        # Spectral centroid (frequency center of mass)
        freqs = np.arange(spikes_bands_spectrogram.shape[0])
        band_energies = np.sum(spikes_bands_spectrogram, axis=1)
        if np.sum(band_energies) > 0:
            spectral_centroid = np.sum(freqs * band_energies) / np.sum(band_energies)
        else:
            spectral_centroid = 0
        features.append(spectral_centroid)
        
        # Spectral spread
        if np.sum(band_energies) > 0:
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * band_energies) / np.sum(band_energies))
        else:
            spectral_spread = 0
        features.append(spectral_spread)
        
        return features


class SCTNFeatureExtractor:
    """
    SCTN-based feature extraction following the proposed approach
    Enhanced version based on successful geophone pattern analysis
    """
    
    def __init__(self):
        self.frequency_bands = {
            'LOW_FREQ': (20, 30),      # Environmental noise
            'CAR_APPROACH': (30, 34),  # Vehicle approach
            'CAR_PEAK': (34, 40),      # Main vehicle signature  
            'CAR_TAIL': (40, 48),      # Vehicle departure
            'MID_GAP': (48, 60),       # Transition band
            'HUMAN_PEAK': (60, 70),    # Primary footstep frequency
            'HUMAN_TAIL': (70, 85),    # Secondary footstep harmonics
            'HIGH_FREQ': (85, 100)     # High frequency noise
        }
        
        # Signal-specific band indices for pattern detection
        self.car_signature_bands = [1, 2, 3]    # 30-48 Hz (car detection)
        self.human_signature_bands = [5, 6]     # 60-85 Hz (human detection)
        self.noise_bands = [0, 7]               # Background noise
        
        print(f"🧠 SCTNFeatureExtractor initialized")
        print(f"   🚗 Car signature bands: {self.car_signature_bands}")
        print(f"   👤 Human signature bands: {self.human_signature_bands}")
    
    def extract_features(self, spikes_bands_spectrogram, duration, signal_type='unknown'):
        """
        Extract SCTN-based features optimized for geophone signals
        Based on successful pattern analysis results
        """
        features = []
        
        if spikes_bands_spectrogram.shape[0] != 8:
            print(f"⚠️ Expected 8 bands, got {spikes_bands_spectrogram.shape[0]}")
            return np.zeros(35)  # Return default features
        
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        # 1. SIGNATURE ENERGY ANALYSIS (Most discriminative features)
        car_energy = np.sum(spikes_bands_spectrogram[self.car_signature_bands, :])
        human_energy = np.sum(spikes_bands_spectrogram[self.human_signature_bands, :])
        noise_energy = np.sum(spikes_bands_spectrogram[self.noise_bands, :])
        total_energy = np.sum(spikes_bands_spectrogram)
        
        # Key discriminative ratios
        car_ratio = car_energy / (total_energy + 1e-10)
        human_ratio = human_energy / (total_energy + 1e-10)
        noise_ratio = noise_energy / (total_energy + 1e-10)
        signal_to_noise = (car_energy + human_energy) / (noise_energy + 1e-10)
        
        features.extend([car_ratio, human_ratio, noise_ratio, signal_to_noise])
        
        # 2. TEMPORAL PATTERN ANALYSIS
        # Car-specific: periodic patterns
        car_temporal = np.sum(spikes_bands_spectrogram[self.car_signature_bands, :], axis=0)
        car_periodicity = self._analyze_periodicity(car_temporal)
        car_consistency = self._analyze_consistency(car_temporal)
        
        # Human-specific: burst patterns  
        human_temporal = np.sum(spikes_bands_spectrogram[self.human_signature_bands, :], axis=0)
        human_burstiness = self._analyze_burstiness(human_temporal)
        human_sparsity = self._analyze_sparsity(human_temporal)
        
        features.extend([car_periodicity, car_consistency, human_burstiness, human_sparsity])
        
        # 3. INDIVIDUAL BAND FEATURES
        for band_idx in range(n_bands):
            band_data = spikes_bands_spectrogram[band_idx, :]
            band_features = self._extract_band_features(band_data)
            features.extend(band_features)
        
        # 4. CROSS-BAND CORRELATIONS
        correlation_features = self._extract_correlation_features(spikes_bands_spectrogram)
        features.extend(correlation_features)
        
        # 5. SIGNAL QUALITY METRICS
        quality_features = self._extract_quality_features(spikes_bands_spectrogram)
        features.extend(quality_features)
        
        return np.array(features)
    
    def _analyze_periodicity(self, signal):
        """Detect periodicity for car signals"""
        if len(signal) < 20:
            return 0
        
        # Autocorrelation analysis
        autocorr = np.correlate(signal, signal, mode='full')
        center = len(autocorr) // 2
        autocorr = autocorr[center:]
        
        # Look for peaks indicating periodicity
        if len(autocorr) > 50:
            periodicity = autocorr[50] / (autocorr[0] + 1e-10)
        else:
            periodicity = 0
        
        return max(0, periodicity)
    
    def _analyze_consistency(self, signal):
        """Analyze temporal consistency"""
        if len(signal) == 0:
            return 0
        
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        consistency = 1.0 / (1.0 + std_val / (mean_val + 1e-10))
        
        return consistency
    
    def _analyze_burstiness(self, signal):
        """Detect burst patterns for human signals"""
        if len(signal) == 0:
            return 0
        
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        above_threshold = signal > threshold
        
        # Count burst transitions
        bursts = np.diff(np.concatenate([[False], above_threshold]))
        burst_count = np.sum(bursts == True)
        burstiness = burst_count / len(signal) * 100
        
        return burstiness
    
    def _analyze_sparsity(self, signal):
        """Analyze signal sparsity"""
        if len(signal) == 0:
            return 0
        
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        active_bins = np.sum(signal > threshold)
        sparsity = 1.0 - (active_bins / len(signal))
        
        return sparsity
    
    def _extract_band_features(self, band_data):
        """Extract features for individual band"""
        if len(band_data) == 0:
            return [0, 0, 0]
        
        features = [
            np.mean(band_data),
            np.std(band_data),
            np.max(band_data)
        ]
        
        return features
    
    def _extract_correlation_features(self, spectrogram):
        """Extract cross-band correlations"""
        features = []
        
        # Key correlations
        car_human_corr = self._calculate_correlation(
            np.sum(spectrogram[self.car_signature_bands, :], axis=0),
            np.sum(spectrogram[self.human_signature_bands, :], axis=0)
        )
        features.append(car_human_corr)
        
        return features
    
    def _extract_quality_features(self, spectrogram):
        """Extract signal quality metrics"""
        features = []
        
        # Activity concentration
        overall_activity = np.sum(spectrogram, axis=0)
        if len(overall_activity) > 0:
            activity_entropy = self._calculate_entropy(overall_activity)
            concentration = 1 / (1 + activity_entropy)
        else:
            concentration = 0
        
        features.append(concentration)
        
        # Peak characteristics
        max_peak = np.max(spectrogram)
        features.append(max_peak)
        
        return features
    
    def _calculate_correlation(self, sig1, sig2):
        """Calculate correlation between two signals"""
        if len(sig1) != len(sig2) or len(sig1) == 0:
            return 0
        
        corr_matrix = np.corrcoef(sig1, sig2)
        if corr_matrix.shape == (2, 2):
            correlation = corr_matrix[0, 1]
            return correlation if not np.isnan(correlation) else 0
        return 0
    
    def _calculate_entropy(self, signal):
        """Calculate entropy of signal distribution"""
        if len(signal) == 0:
            return 0
        
        hist, _ = np.histogram(signal, bins=10)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log(hist))
        
        return entropy


class GeophoneClassificationSystem:
    """
    Comprehensive geophone classification system following EEG study methodology
    """
    
    def __init__(self):
        self.fft_extractor = FFTFeatureExtractor()
        self.sctn_extractor = SCTNFeatureExtractor()
        self.scalers = {}
        
        # Classification models (matching EEG study)
        self.models = {
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5),
        }
        
        print(f"🔬 GeophoneClassificationSystem initialized")
        print(f"   🤖 {len(self.models)} classification models")
    
    def load_chunk_data(self, chunks_base_dir):
        """Load saved chunk data"""
        print(f"🔄 Loading chunks from {chunks_base_dir}")
        
        chunk_data = {}
        
        for signal_type in ['car', 'car_nothing', 'human', 'human_nothing']:
            chunk_dir = os.path.join(chunks_base_dir, signal_type)
            index_file = os.path.join(chunk_dir, 'chunk_index.pkl')
            
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    chunk_index = pickle.load(f)
                
                chunks = []
                for chunk_file in chunk_index['chunk_files']:
                    if os.path.exists(chunk_file):
                        with open(chunk_file, 'rb') as f:
                            chunk = pickle.load(f)
                        chunks.append(chunk)
                
                chunk_data[signal_type] = chunks
                print(f"   ✅ {signal_type}: {len(chunks)} chunks")
        
        return chunk_data
    
    def extract_features_from_chunks(self, chunk_data, feature_type='both'):
        """
        Extract features from chunks following EEG study methodology
        feature_type: 'fft', 'sctn', or 'both'
        """
        print(f"\n🔬 EXTRACTING {feature_type.upper()} FEATURES FROM CHUNKS")
        print("-" * 50)
        
        all_fft_features = []
        all_sctn_features = []
        all_labels = []
        chunk_info = []
        
        for signal_type, chunks in chunk_data.items():
            # Determine signal class and task type
            if signal_type.startswith('car'):
                task_type = 'car'
                is_positive = not signal_type.endswith('_nothing')
            else:  # human
                task_type = 'human'
                is_positive = not signal_type.endswith('_nothing')
            
            label = 1 if is_positive else 0  # 1=signal present, 0=nothing
            
            print(f"📊 Processing {signal_type} ({len(chunks)} chunks)...")
            
            for chunk_idx, chunk in enumerate(chunks):
                if 'spikes_bands_spectrogram' not in chunk:
                    continue
                
                spikes_bands_spectrogram = chunk['spikes_bands_spectrogram']
                duration = chunk.get('duration', 120)
                
                # Extract FFT features
                if feature_type in ['fft', 'both']:
                    fft_features = self.fft_extractor.extract_features(
                        spikes_bands_spectrogram, duration
                    )
                    all_fft_features.append(fft_features)
                
                # Extract SCTN features
                if feature_type in ['sctn', 'both']:
                    sctn_features = self.sctn_extractor.extract_features(
                        spikes_bands_spectrogram, duration, task_type
                    )
                    all_sctn_features.append(sctn_features)
                
                all_labels.append(label)
                chunk_info.append({
                    'signal_type': signal_type,
                    'chunk_idx': chunk_idx,
                    'task_type': task_type,
                    'label': label
                })
        
        # Convert to arrays
        if feature_type == 'fft':
            return np.array(all_fft_features), np.array(all_labels), chunk_info
        elif feature_type == 'sctn':
            return np.array(all_sctn_features), np.array(all_labels), chunk_info
        else:  # both
            return (np.array(all_fft_features), np.array(all_sctn_features), 
                    np.array(all_labels), chunk_info)
    
    def create_task_specific_datasets(self, features_fft, features_sctn, labels, chunk_info):
        """
        Create task-specific datasets for car detection and human detection
        Following EEG study approach with different tasks
        """
        datasets = {}
        
        # Car detection task: car vs car_nothing
        car_indices = [i for i, info in enumerate(chunk_info) if info['task_type'] == 'car']
        if car_indices:
            datasets['car_detection'] = {
                'fft_features': features_fft[car_indices],
                'sctn_features': features_sctn[car_indices],
                'labels': labels[car_indices],
                'chunk_info': [chunk_info[i] for i in car_indices]
            }
        
        # Human detection task: human vs human_nothing  
        human_indices = [i for i, info in enumerate(chunk_info) if info['task_type'] == 'human']
        if human_indices:
            datasets['human_detection'] = {
                'fft_features': features_fft[human_indices],
                'sctn_features': features_sctn[human_indices],
                'labels': labels[human_indices],
                'chunk_info': [chunk_info[i] for i in human_indices]
            }
        
        return datasets
    
    def evaluate_models(self, datasets):
        """
        Evaluate models following EEG study methodology
        Compare FFT vs SCTN features across multiple models
        """
        print(f"\n🧪 MODEL EVALUATION FOLLOWING EEG STUDY METHODOLOGY")
        print("=" * 60)
        
        results = {}
        
        for task_name, dataset in datasets.items():
            print(f"\n📊 TASK: {task_name.upper()}")
            print("-" * 40)
            
            results[task_name] = {}
            
            # Get data
            X_fft = dataset['fft_features']
            X_sctn = dataset['sctn_features']
            y = dataset['labels']
            
            print(f"Dataset: {len(y)} samples, {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
            
            # Evaluate each model with both feature types
            for model_name, model in self.models.items():
                results[task_name][model_name] = {}
                
                # FFT features
                fft_accuracy = self._evaluate_single_model(model, X_fft, y, f"FFT_{model_name}")
                results[task_name][model_name]['FFT'] = fft_accuracy
                
                # SCTN features
                sctn_accuracy = self._evaluate_single_model(model, X_sctn, y, f"SCTN_{model_name}")
                results[task_name][model_name]['SCTN'] = sctn_accuracy
                
                # Calculate improvement
                improvement = sctn_accuracy - fft_accuracy
                results[task_name][model_name]['Improvement'] = improvement
                
                print(f"{model_name:15} | FFT: {fft_accuracy:.1%} | SCTN: {sctn_accuracy:.1%} | Δ: {improvement:+.1%}")
        
        return results
    
    def _evaluate_single_model(self, model, X, y, model_name):
        """Evaluate single model with cross-validation"""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation (like EEG study)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            
            return np.mean(scores)
            
        except Exception as e:
            print(f"⚠️ Error evaluating {model_name}: {e}")
            return 0.0
    
    def create_weighted_ensemble(self, X, y):
        """Create weighted ensemble classifier (best performer in EEG study)"""
        # Create ensemble of best models
        ensemble = VotingClassifier(
            estimators=[
                ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
            ],
            voting='soft'
        )
        
        return ensemble
    
    def generate_comparison_table(self, results):
        """Generate comparison table like in EEG study"""
        print(f"\n📊 CLASSIFICATION RESULTS COMPARISON")
        print("=" * 60)
        print("Following EEG Study Table Format")
        print()
        
        # Create results table
        print(f"{'Model':<15} {'Task':<15} {'FFT':<8} {'SCTN':<8} {'Improvement':<12}")
        print("-" * 60)
        
        for task_name, task_results in results.items():
            for model_name, model_results in task_results.items():
                fft_acc = model_results['FFT']
                sctn_acc = model_results['SCTN']
                improvement = model_results['Improvement']
                
                print(f"{model_name:<15} {task_name:<15} {fft_acc:<8.1%} {sctn_acc:<8.1%} {improvement:<+12.1%}")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS")
        print("-" * 30)
        
        all_improvements = []
        all_fft_scores = []
        all_sctn_scores = []
        
        for task_results in results.values():
            for model_results in task_results.values():
                all_fft_scores.append(model_results['FFT'])
                all_sctn_scores.append(model_results['SCTN'])
                all_improvements.append(model_results['Improvement'])
        
        print(f"Average FFT Accuracy:    {np.mean(all_fft_scores):.1%}")
        print(f"Average SCTN Accuracy:   {np.mean(all_sctn_scores):.1%}")
        print(f"Average Improvement:     {np.mean(all_improvements):+.1%}")
        print(f"Max Improvement:         {np.max(all_improvements):+.1%}")
        print(f"SCTN Better in:          {np.sum(np.array(all_improvements) > 0)}/{len(all_improvements)} cases")


def main():
    """Main execution following EEG study methodology"""
    print("🚀 COMPREHENSIVE GEOPHONE CLASSIFICATION SYSTEM")
    print("=" * 60)
    print("Following EEG Study Methodology: FFT vs SCTN Feature Comparison")
    print()
    
    # Initialize system
    system = GeophoneClassificationSystem()
    
    # Load chunk data
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    chunk_data = system.load_chunk_data(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return
    
    # Extract features (both FFT and SCTN)
    features_fft, features_sctn, labels, chunk_info = system.extract_features_from_chunks(
        chunk_data, feature_type='both'
    )
    
    print(f"\n📊 FEATURE EXTRACTION SUMMARY")
    print(f"   Total chunks processed: {len(labels)}")
    print(f"   FFT features per chunk: {features_fft.shape[1]}")
    print(f"   SCTN features per chunk: {features_sctn.shape[1]}")
    print(f"   Positive samples: {np.sum(labels)}")
    print(f"   Negative samples: {len(labels) - np.sum(labels)}")
    
    # Create task-specific datasets
    datasets = system.create_task_specific_datasets(
        features_fft, features_sctn, labels, chunk_info
    )
    
    print(f"\n🎯 TASK-SPECIFIC DATASETS")
    for task_name, dataset in datasets.items():
        n_samples = len(dataset['labels'])
        n_positive = np.sum(dataset['labels'])
        print(f"   {task_name}: {n_samples} samples ({n_positive} positive, {n_samples - n_positive} negative)")
    
    # Evaluate models
    results = system.evaluate_models(datasets)
    
    # Generate comparison table
    system.generate_comparison_table(results)
    
    # Additional analysis
    print(f"\n🔍 DETAILED ANALYSIS")
    print("-" * 30)
    
    best_improvements = []
    for task_name, task_results in results.items():
        task_improvements = [model_results['Improvement'] for model_results in task_results.values()]
        best_improvement = max(task_improvements)
        best_model = max(task_results.keys(), key=lambda k: task_results[k]['Improvement'])
        
        print(f"{task_name}: Best improvement {best_improvement:+.1%} with {best_model}")
        best_improvements.append(best_improvement)
    
    print(f"\nOverall best improvement: {max(best_improvements):+.1%}")
    
    if np.mean([r for task in results.values() for r in [m['Improvement'] for m in task.values()]]) > 0:
        print("🎉 SCTN features outperform FFT features on average!")
    else:
        print("📊 Mixed results - task-dependent performance")
    
    return results


if __name__ == "__main__":
    results = main() 