#!/usr/bin/env python3
"""
Advanced Geophone Footprint Analyzer
Based on EEG Study Methodology and Resonator Pattern Analysis

This system implements:
1. Raw data FFT analysis for frequency domain features
2. Chunk-based SCTN analysis for temporal spike patterns
3. Footprint detection using supervised STDP approaches
4. Pattern extraction similar to mental attention detection
5. Comprehensive feature engineering for optimal classification

Key insights from old notebook examples:
- EEG mental attention: Pattern detection across frequency bands
- Supervised STDP: Learning optimal resonator parameters for signatures
- Range STDP: Temporal pattern analysis with phase detection
- Semi-supervised: Burst detection and activity patterns
"""

import numpy as np
import pandas as pd
import os
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import welch, spectrogram, periodogram, find_peaks
import warnings
warnings.filterwarnings('ignore')

# Frequency bands based on analysis
FREQUENCY_BANDS = {
    'LOW_FREQ': (20, 30),      # Background/noise
    'CAR_APPROACH': (30, 34),  # Car approach signature
    'CAR_PEAK': (34, 40),      # Main car frequency
    'CAR_TAIL': (40, 48),      # Car departure signature
    'MID_GAP': (48, 60),       # Transition zone
    'HUMAN_PEAK': (60, 70),    # Primary human footstep frequency
    'HUMAN_TAIL': (70, 85),    # Secondary human harmonics
    'HIGH_FREQ': (85, 100)     # High frequency activity
}

# Resonator parameters from old notebook analysis
CLK_FREQ = 153600
RESONATOR_FREQS = [10.5, 11.5, 12.8, 15.8, 16.6, 19.4, 22.0, 24.8, 28.4, 30.5, 34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2]

class AdvancedGeophoneAnalyzer:
    """
    Advanced analyzer combining FFT and SCTN approaches for optimal footprint detection
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.frequency_bands = FREQUENCY_BANDS
        self.clk_freq = CLK_FREQ
        self.resonator_freqs = RESONATOR_FREQS
        
    def load_raw_data(self, data_path):
        """Load raw time series data from CSV files"""
        print(f"📄 Loading raw data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Extract time and amplitude
        time_series = data['time_s'].values
        amplitude = data['amplitude'].values
        
        # Calculate sampling rate
        dt = np.mean(np.diff(time_series))
        fs = 1.0 / dt
        
        print(f"   📊 Loaded {len(amplitude)} samples")
        print(f"   ⏱️ Duration: {time_series[-1]:.2f}s")
        print(f"   📡 Sampling rate: {fs:.1f} Hz")
        
        return time_series, amplitude, fs
    
    def load_chunks(self, chunks_dir, signal_type):
        """Load pre-processed chunks with resonator outputs"""
        print(f"📦 Loading {signal_type} chunks from {chunks_dir}")
        
        index_file = os.path.join(chunks_dir, signal_type, 'chunk_index.pkl')
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Chunk index not found: {index_file}")
        
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        chunks = []
        for chunk_file in chunk_index['chunk_files']:
            if os.path.exists(chunk_file):
                with open(chunk_file, 'rb') as f:
                    chunk = pickle.load(f)
                chunks.append(chunk)
        
        print(f"   📦 Loaded {len(chunks)} chunks")
        if chunks:
            print(f"   ⏱️ Chunk duration: {chunks[0]['duration']}s")
            print(f"   📊 Spikegram shape: {chunks[0]['spikes_bands_spectrogram'].shape}")
        
        return chunks
    
    def extract_fft_features(self, time_series, amplitude, fs, segment_duration=30):
        """
        Extract FFT-based frequency domain features from raw signal
        Based on EEG analysis approach
        """
        print(f"🌊 Extracting FFT features (segment duration: {segment_duration}s)")
        
        features_list = []
        labels_list = []
        
        # Segment the signal
        segment_samples = int(segment_duration * fs)
        n_segments = len(amplitude) // segment_samples
        
        for i in range(n_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            
            segment = amplitude[start_idx:end_idx]
            segment_time = time_series[start_idx:end_idx]
            
            # Extract comprehensive FFT features
            features = self._extract_segment_fft_features(segment, fs)
            features_list.append(features)
        
        print(f"   📊 Extracted {len(features_list)} FFT segments")
        print(f"   🎯 Features per segment: {len(features_list[0]) if features_list else 0}")
        
        return np.array(features_list)
    
    def _extract_segment_fft_features(self, segment, fs):
        """Extract comprehensive FFT features from a signal segment"""
        features = []
        
        # 1. Power Spectral Density using Welch's method
        freqs, psd = welch(segment, fs, nperseg=min(len(segment)//4, 512))
        
        # 2. Band power features
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.sum(psd[band_mask])
                band_mean_freq = np.sum(freqs[band_mask] * psd[band_mask]) / (np.sum(psd[band_mask]) + 1e-10)
                band_peak_freq = freqs[band_mask][np.argmax(psd[band_mask])]
                band_peak_power = np.max(psd[band_mask])
                
                features.extend([
                    band_power,
                    band_mean_freq,
                    band_peak_freq, 
                    band_peak_power,
                    np.std(psd[band_mask]),
                    skew(psd[band_mask]),
                    kurtosis(psd[band_mask])
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # 3. Cross-band ratios (key discriminative features)
        total_power = np.sum(psd)
        car_power = np.sum(psd[(freqs >= 30) & (freqs <= 48)])
        human_power = np.sum(psd[(freqs >= 60) & (freqs <= 85)])
        
        features.extend([
            car_power / (total_power + 1e-10),    # Car dominance ratio
            human_power / (total_power + 1e-10),  # Human dominance ratio
            car_power / (human_power + 1e-10),    # Car vs human ratio
            (car_power + human_power) / (total_power + 1e-10)  # Signal vs noise ratio
        ])
        
        # 4. Spectral characteristics
        features.extend([
            np.sum(psd),                          # Total power
            freqs[np.argmax(psd)],               # Peak frequency
            np.max(psd),                         # Peak power
            np.mean(freqs),                      # Mean frequency
            np.median(freqs),                    # Median frequency
            np.std(psd),                         # Spectral spread
            skew(psd),                           # Spectral skewness
            kurtosis(psd)                        # Spectral kurtosis
        ])
        
        # 5. Temporal features in frequency domain
        # Spectral rolloff, centroid, bandwidth
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / (total_power + 1e-10)
        
        # Spectral rolloff (frequency below which 85% of power is contained)
        rolloff_threshold = 0.85 * total_power
        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / (total_power + 1e-10))
        
        features.extend([
            spectral_centroid,
            spectral_rolloff,
            spectral_bandwidth
        ])
        
        return features
    
    def extract_sctn_features(self, chunks):
        """
        Extract SCTN-based features from processed chunks
        Based on supervised STDP and range analysis approaches
        """
        print(f"🧠 Extracting SCTN features from {len(chunks)} chunks")
        
        features_list = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Extract features from spikegram and resonator outputs
            features = self._extract_chunk_sctn_features(chunk)
            features_list.append(features)
        
        print(f"   🎯 Features per chunk: {len(features_list[0]) if features_list else 0}")
        
        return np.array(features_list)
    
    def _extract_chunk_sctn_features(self, chunk):
        """Extract comprehensive SCTN features from a single chunk"""
        features = []
        
        # 1. Spikegram pattern analysis
        spikegram = chunk['spikes_bands_spectrogram']  # Shape: (8, 12000)
        n_bands, n_time_bins = spikegram.shape
        
        # Band-wise features
        for band_idx in range(n_bands):
            band_data = spikegram[band_idx]
            
            # Basic statistics
            features.extend([
                np.mean(band_data),
                np.std(band_data),
                np.max(band_data),
                np.min(band_data),
                skew(band_data),
                kurtosis(band_data)
            ])
            
            # Activity patterns
            active_bins = band_data > 0
            activity_ratio = np.sum(active_bins) / len(band_data)
            
            # Burst detection (similar to EEG approach)
            threshold = np.mean(band_data) + 2 * np.std(band_data)
            bursts = band_data > threshold
            n_bursts = self._count_bursts(bursts)
            burst_ratio = n_bursts / len(band_data) * 100
            
            # Periodicity detection (key for car detection)
            periodicity_score = self._detect_periodicity(band_data)
            
            features.extend([
                activity_ratio,
                n_bursts,
                burst_ratio,
                periodicity_score
            ])
        
        # 2. Cross-band correlations and interactions
        # Car-specific patterns (bands 1-4: 30-48 Hz)
        car_bands = spikegram[1:5]
        car_pattern = np.mean(car_bands, axis=0)
        car_consistency = np.std([np.corrcoef(car_bands[i], car_pattern)[0,1] for i in range(len(car_bands))])
        
        # Human-specific patterns (bands 5-7: 60-85 Hz) 
        human_bands = spikegram[5:8]
        human_pattern = np.mean(human_bands, axis=0)
        human_burstiness = self._calculate_burstiness(human_pattern)
        
        # Pattern separation
        car_strength = np.mean(car_pattern)
        human_strength = np.mean(human_pattern)
        pattern_separation = abs(car_strength - human_strength)
        
        features.extend([
            car_consistency,
            human_burstiness,
            car_strength,
            human_strength,
            pattern_separation,
            car_strength / (human_strength + 1e-10),  # Car vs human ratio
        ])
        
        # 3. Temporal dynamics analysis
        # Overall activity concentration
        total_activity = np.sum(spikegram, axis=0)
        activity_peaks = find_peaks(total_activity, height=np.mean(total_activity))[0]
        
        features.extend([
            len(activity_peaks),
            np.std(total_activity),
            np.max(total_activity),
            self._calculate_temporal_regularity(total_activity)
        ])
        
        # 4. Resonator output analysis (using spike timestamps)
        resonator_outputs = chunk['resonator_outputs'][self.clk_freq]
        
        # Resonator-specific features
        for resonator_idx in range(min(len(resonator_outputs), 8)):  # First 8 resonators
            spike_times = resonator_outputs[resonator_idx]
            
            if len(spike_times) > 0:
                # Convert spike times to rates
                duration = chunk['duration']
                spike_rate = len(spike_times) / duration
                
                # Inter-spike interval analysis
                if len(spike_times) > 1:
                    isis = np.diff(spike_times)
                    isi_mean = np.mean(isis)
                    isi_std = np.std(isis)
                    isi_cv = isi_std / (isi_mean + 1e-10)  # Coefficient of variation
                else:
                    isi_mean = isi_std = isi_cv = 0
                
                features.extend([
                    spike_rate,
                    isi_mean,
                    isi_std,
                    isi_cv
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        return features
    
    def _count_bursts(self, binary_activity):
        """Count number of burst events in binary activity pattern"""
        if len(binary_activity) == 0:
            return 0
        
        # Count transitions from inactive to active
        diff = np.diff(np.concatenate([[False], binary_activity, [False]]).astype(int))
        return np.sum(diff == 1)
    
    def _detect_periodicity(self, signal):
        """Detect periodicity in signal using autocorrelation"""
        if len(signal) < 10:
            return 0
        
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        if len(autocorr) > 10:
            peaks = find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))[0]
            return len(peaks) / len(autocorr) * 100
        
        return 0
    
    def _calculate_burstiness(self, signal):
        """Calculate burstiness metric for human footstep detection"""
        if len(signal) == 0:
            return 0
        
        # Identify high-activity periods
        threshold = np.mean(signal) + np.std(signal)
        high_activity = signal > threshold
        
        # Calculate burstiness as ratio of high-activity variance to mean
        if np.mean(high_activity.astype(float)) > 0:
            return np.var(high_activity.astype(float)) / (np.mean(high_activity.astype(float)) + 1e-10)
        
        return 0
    
    def _calculate_temporal_regularity(self, signal):
        """Calculate temporal regularity of activity pattern"""
        if len(signal) < 10:
            return 0
        
        # Find peaks
        peaks = find_peaks(signal, height=np.mean(signal))[0]
        
        if len(peaks) > 2:
            # Calculate peak intervals
            intervals = np.diff(peaks)
            # Regularity is inverse of interval variance
            return 1.0 / (np.std(intervals) + 1e-10)
        
        return 0
    
    def create_footprint_features(self, raw_data_path, chunks_dir, signal_type):
        """
        Create comprehensive footprint features combining FFT and SCTN approaches
        """
        print(f"\n🔍 EXTRACTING FOOTPRINT FEATURES FOR {signal_type.upper()}")
        print("=" * 60)
        
        all_features = []
        
        # 1. Extract FFT features from raw data
        try:
            time_series, amplitude, fs = self.load_raw_data(raw_data_path)
            fft_features = self.extract_fft_features(time_series, amplitude, fs)
            print(f"✅ FFT features: {fft_features.shape}")
        except Exception as e:
            print(f"❌ FFT extraction failed: {e}")
            fft_features = np.array([])
        
        # 2. Extract SCTN features from chunks
        try:
            chunks = self.load_chunks(chunks_dir, signal_type)
            sctn_features = self.extract_sctn_features(chunks)
            print(f"✅ SCTN features: {sctn_features.shape}")
        except Exception as e:
            print(f"❌ SCTN extraction failed: {e}")
            sctn_features = np.array([])
        
        # 3. Combine features with intelligent alignment
        if fft_features.size > 0 and sctn_features.size > 0:
            # Align features by resampling to common length
            min_samples = min(len(fft_features), len(sctn_features))
            
            # Resample FFT features
            if len(fft_features) > min_samples:
                fft_indices = np.linspace(0, len(fft_features)-1, min_samples).astype(int)
                fft_features = fft_features[fft_indices]
            
            # Resample SCTN features
            if len(sctn_features) > min_samples:
                sctn_indices = np.linspace(0, len(sctn_features)-1, min_samples).astype(int)
                sctn_features = sctn_features[sctn_indices]
            
            # Combine features
            all_features = np.hstack([fft_features, sctn_features])
            print(f"🔗 Combined features: {all_features.shape}")
            
        elif fft_features.size > 0:
            all_features = fft_features
            print(f"⚠️ Using FFT features only: {all_features.shape}")
            
        elif sctn_features.size > 0:
            all_features = sctn_features
            print(f"⚠️ Using SCTN features only: {all_features.shape}")
        
        return all_features
    
    def train_comprehensive_classifier(self, features_dict, test_size=0.25):
        """
        Train comprehensive classifier using combined features
        Following EEG study methodology with multiple algorithms
        """
        print(f"\n🎯 TRAINING COMPREHENSIVE CLASSIFIER")
        print("=" * 50)
        
        # Prepare dataset
        X, y = self._prepare_training_data(features_dict)
        
        if len(X) == 0:
            print("❌ No training data available")
            return None
        
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"📈 Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define multiple classifiers (EEG study approach)
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        results = {}
        
        # Train and evaluate each classifier
        print(f"\n📋 CLASSIFIER COMPARISON")
        print("-" * 50)
        
        for name, clf in classifiers.items():
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = clf.score(X_train_scaled, y_train)
            test_acc = clf.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'classifier': clf,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            print(f"{name:<20} Train: {train_acc:.1%}  Test: {test_acc:.1%}  CV: {cv_mean:.1%} ± {cv_std:.1%}")
        
        # Create ensemble classifier
        best_classifiers = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
        ensemble_clfs = [(name, results[name]['classifier']) for name, _ in best_classifiers]
        
        ensemble = VotingClassifier(estimators=ensemble_clfs, voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        
        ensemble_acc = ensemble.score(X_test_scaled, y_test)
        print(f"\n🏆 Ensemble Accuracy: {ensemble_acc:.1%}")
        
        # Detailed evaluation of best classifier
        best_name = best_classifiers[0][0]
        best_clf = results[best_name]['classifier']
        
        y_pred = best_clf.predict(X_test_scaled)
        
        print(f"\n📊 DETAILED RESULTS ({best_name})")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=['Signal', 'Nothing']))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'best_classifier': best_clf,
            'ensemble': ensemble,
            'scaler': self.scaler,
            'results': results,
            'test_accuracy': results[best_name]['test_accuracy'],
            'feature_shape': X.shape[1]
        }
    
    def _prepare_training_data(self, features_dict):
        """Prepare training data from features dictionary"""
        X = []
        y = []
        
        for signal_type, features in features_dict.items():
            if features.size == 0:
                continue
                
            # Label encoding: signal=0, nothing=1
            if 'nothing' in signal_type:
                labels = np.ones(len(features))
            else:
                labels = np.zeros(len(features))
            
            X.append(features)
            y.append(labels)
        
        if X:
            X = np.vstack(X)
            y = np.concatenate(y)
            return X, y.astype(int)
        
        return np.array([]), np.array([])

def main():
    """Main execution function"""
    print("🚀 ADVANCED GEOPHONE FOOTPRINT ANALYZER")
    print("Following EEG Study & Resonator Analysis Methodology")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AdvancedGeophoneAnalyzer()
    
    # Define data paths
    data_dir = "project/data"
    chunks_dir = "project/MyCode/chunked_output"
    
    data_files = {
        'car': f"{data_dir}/car.csv",
        'car_nothing': f"{data_dir}/car_nothing.csv", 
        'human': f"{data_dir}/human.csv",
        'human_nothing': f"{data_dir}/human_nothing.csv"
    }
    
    # Extract features for each signal type
    features_dict = {}
    
    for signal_type, data_file in data_files.items():
        if os.path.exists(data_file):
            features = analyzer.create_footprint_features(data_file, chunks_dir, signal_type)
            features_dict[signal_type] = features
        else:
            print(f"⚠️ Data file not found: {data_file}")
            features_dict[signal_type] = np.array([])
    
    # Train car vs car_nothing classifier
    if features_dict['car'].size > 0 or features_dict['car_nothing'].size > 0:
        print(f"\n🚗 TRAINING CAR CLASSIFIER")
        car_features = {
            'car': features_dict['car'],
            'car_nothing': features_dict['car_nothing']
        }
        car_results = analyzer.train_comprehensive_classifier(car_features)
        
        if car_results:
            print(f"✅ Car classifier trained: {car_results['test_accuracy']:.1%} accuracy")
    
    # Train human vs human_nothing classifier  
    if features_dict['human'].size > 0 or features_dict['human_nothing'].size > 0:
        print(f"\n👤 TRAINING HUMAN CLASSIFIER")
        human_features = {
            'human': features_dict['human'],
            'human_nothing': features_dict['human_nothing']
        }
        human_results = analyzer.train_comprehensive_classifier(human_features)
        
        if human_results:
            print(f"✅ Human classifier trained: {human_results['test_accuracy']:.1%} accuracy")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("Advanced footprint detection system ready for deployment.")

if __name__ == "__main__":
    main() 