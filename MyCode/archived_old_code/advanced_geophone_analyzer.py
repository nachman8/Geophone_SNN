#!/usr/bin/env python3
"""
Advanced Geophone Analyzer
Combines FFT features from raw data and SCTN features from chunks
Based on EEG study methodology and resonator pattern analysis
"""

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
import warnings
warnings.filterwarnings('ignore')

# Frequency bands for analysis
FREQUENCY_BANDS = {
    'LOW_FREQ': (20, 30),
    'CAR_APPROACH': (30, 34), 
    'CAR_PEAK': (34, 40),
    'CAR_TAIL': (40, 48),
    'MID_GAP': (48, 60),
    'HUMAN_PEAK': (60, 70),
    'HUMAN_TAIL': (70, 85),
    'HIGH_FREQ': (85, 100)
}

class AdvancedGeophoneAnalyzer:
    """Advanced analyzer for footprint detection and pattern analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.frequency_bands = FREQUENCY_BANDS
        
    def load_raw_data(self, data_path):
        """Load raw time series data from CSV"""
        print(f"📄 Loading raw data from {data_path}")
        data = pd.read_csv(data_path)
        
        time_series = data['time_s'].values
        amplitude = data['amplitude'].values
        
        dt = np.mean(np.diff(time_series))
        fs = 1.0 / dt
        
        print(f"   📊 {len(amplitude)} samples, {time_series[-1]:.2f}s, {fs:.1f}Hz")
        return time_series, amplitude, fs
    
    def load_chunks(self, chunks_dir, signal_type):
        """Load processed chunks with resonator outputs"""
        print(f"📦 Loading {signal_type} chunks")
        
        index_file = os.path.join(chunks_dir, signal_type, 'chunk_index.pkl')
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        chunks = []
        for chunk_file in chunk_index['chunk_files']:
            if os.path.exists(chunk_file):
                with open(chunk_file, 'rb') as f:
                    chunk = pickle.load(f)
                chunks.append(chunk)
        
        print(f"   📦 {len(chunks)} chunks loaded")
        return chunks
    
    def extract_fft_features(self, time_series, amplitude, fs, segment_duration=30):
        """Extract FFT features from raw signal segments"""
        print(f"🌊 Extracting FFT features")
        
        features_list = []
        segment_samples = int(segment_duration * fs)
        n_segments = len(amplitude) // segment_samples
        
        for i in range(n_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            segment = amplitude[start_idx:end_idx]
            
            features = self._extract_segment_fft_features(segment, fs)
            features_list.append(features)
        
        print(f"   📊 {len(features_list)} FFT segments, {len(features_list[0])} features each")
        return np.array(features_list)
    
    def _extract_segment_fft_features(self, segment, fs):
        """Extract comprehensive FFT features from segment"""
        features = []
        
        # Power spectral density
        freqs, psd = welch(segment, fs, nperseg=min(len(segment)//4, 512))
        
        # Band power features
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.sum(psd[band_mask])
                band_peak_freq = freqs[band_mask][np.argmax(psd[band_mask])]
                band_peak_power = np.max(psd[band_mask])
                features.extend([band_power, band_peak_freq, band_peak_power])
            else:
                features.extend([0, 0, 0])
        
        # Cross-band ratios
        total_power = np.sum(psd)
        car_power = np.sum(psd[(freqs >= 30) & (freqs <= 48)])
        human_power = np.sum(psd[(freqs >= 60) & (freqs <= 85)])
        
        features.extend([
            car_power / (total_power + 1e-10),
            human_power / (total_power + 1e-10),
            car_power / (human_power + 1e-10)
        ])
        
        # Spectral characteristics
        features.extend([
            np.sum(psd),
            freqs[np.argmax(psd)],
            np.max(psd),
            np.std(psd)
        ])
        
        return features
    
    def extract_sctn_features(self, chunks):
        """Extract SCTN features from chunks"""
        print(f"🧠 Extracting SCTN features from {len(chunks)} chunks")
        
        features_list = []
        for chunk in chunks:
            features = self._extract_chunk_features(chunk)
            features_list.append(features)
        
        print(f"   🎯 {len(features_list[0])} SCTN features per chunk")
        return np.array(features_list)
    
    def _extract_chunk_features(self, chunk):
        """Extract features from single chunk"""
        features = []
        
        # Spikegram analysis
        spikegram = chunk['spikes_bands_spectrogram']
        n_bands, n_time_bins = spikegram.shape
        
        # Band-wise features
        for band_idx in range(n_bands):
            band_data = spikegram[band_idx]
            
            features.extend([
                np.mean(band_data),
                np.std(band_data),
                np.max(band_data),
                skew(band_data),
                kurtosis(band_data)
            ])
            
            # Activity patterns
            active_ratio = np.sum(band_data > 0) / len(band_data)
            
            # Burst detection
            threshold = np.mean(band_data) + 2 * np.std(band_data)
            bursts = band_data > threshold
            n_bursts = self._count_bursts(bursts)
            
            # Periodicity
            periodicity = self._detect_periodicity(band_data)
            
            features.extend([active_ratio, n_bursts, periodicity])
        
        # Cross-band patterns
        car_bands = spikegram[1:5]  # 30-48 Hz
        human_bands = spikegram[5:8]  # 60-85 Hz
        
        car_strength = np.mean(car_bands)
        human_strength = np.mean(human_bands)
        pattern_separation = abs(car_strength - human_strength)
        
        features.extend([
            car_strength,
            human_strength, 
            pattern_separation,
            car_strength / (human_strength + 1e-10)
        ])
        
        # Temporal dynamics
        total_activity = np.sum(spikegram, axis=0)
        activity_peaks = find_peaks(total_activity, height=np.mean(total_activity))[0]
        
        features.extend([
            len(activity_peaks),
            np.std(total_activity),
            np.max(total_activity)
        ])
        
        return features
    
    def _count_bursts(self, binary_activity):
        """Count burst events"""
        if len(binary_activity) == 0:
            return 0
        diff = np.diff(np.concatenate([[False], binary_activity, [False]]).astype(int))
        return np.sum(diff == 1)
    
    def _detect_periodicity(self, signal):
        """Detect periodicity using autocorrelation"""
        if len(signal) < 10:
            return 0
        
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 10:
            peaks = find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))[0]
            return len(peaks) / len(autocorr) * 100
        return 0
    
    def create_combined_features(self, raw_data_path, chunks_dir, signal_type):
        """Create combined FFT + SCTN features"""
        print(f"\n🔍 PROCESSING {signal_type.upper()}")
        print("=" * 40)
        
        all_features = []
        
        # FFT features from raw data
        try:
            time_series, amplitude, fs = self.load_raw_data(raw_data_path)
            fft_features = self.extract_fft_features(time_series, amplitude, fs)
            print(f"✅ FFT features: {fft_features.shape}")
        except Exception as e:
            print(f"❌ FFT failed: {e}")
            fft_features = np.array([])
        
        # SCTN features from chunks
        try:
            chunks = self.load_chunks(chunks_dir, signal_type)
            sctn_features = self.extract_sctn_features(chunks)
            print(f"✅ SCTN features: {sctn_features.shape}")
        except Exception as e:
            print(f"❌ SCTN failed: {e}")
            sctn_features = np.array([])
        
        # Combine features
        if fft_features.size > 0 and sctn_features.size > 0:
            min_samples = min(len(fft_features), len(sctn_features))
            
            if len(fft_features) > min_samples:
                indices = np.linspace(0, len(fft_features)-1, min_samples).astype(int)
                fft_features = fft_features[indices]
            
            if len(sctn_features) > min_samples:
                indices = np.linspace(0, len(sctn_features)-1, min_samples).astype(int)
                sctn_features = sctn_features[indices]
            
            all_features = np.hstack([fft_features, sctn_features])
            print(f"🔗 Combined: {all_features.shape}")
            
        elif fft_features.size > 0:
            all_features = fft_features
            print(f"⚠️ FFT only: {all_features.shape}")
            
        elif sctn_features.size > 0:
            all_features = sctn_features  
            print(f"⚠️ SCTN only: {all_features.shape}")
        
        return all_features
    
    def train_classifier(self, features_dict, task_name):
        """Train classifier for binary task"""
        print(f"\n🎯 TRAINING {task_name} CLASSIFIER")
        print("=" * 40)
        
        # Prepare data
        X, y = self._prepare_data(features_dict)
        
        if len(X) == 0:
            print("❌ No data available")
            return None
        
        print(f"📊 {len(X)} samples, {X.shape[1]} features")
        print(f"📈 Classes: {np.bincount(y)}")
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Multiple classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        results = {}
        print(f"\n📋 CLASSIFIER COMPARISON")
        
        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, y_train)
            test_acc = clf.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'classifier': clf,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name:<15} Test: {test_acc:.1%}  CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        
        # Best classifier
        best_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        best_clf = results[best_name]['classifier']
        
        y_pred = best_clf.predict(X_test_scaled)
        print(f"\n📊 BEST: {best_name} ({results[best_name]['test_accuracy']:.1%})")
        print(classification_report(y_test, y_pred, target_names=['Signal', 'Nothing']))
        
        return {
            'classifier': best_clf,
            'scaler': self.scaler,
            'accuracy': results[best_name]['test_accuracy'],
            'results': results
        }
    
    def _prepare_data(self, features_dict):
        """Prepare training data"""
        X, y = [], []
        
        for signal_type, features in features_dict.items():
            if features.size == 0:
                continue
            
            labels = np.ones(len(features)) if 'nothing' in signal_type else np.zeros(len(features))
            X.append(features)
            y.append(labels)
        
        if X:
            return np.vstack(X), np.concatenate(y).astype(int)
        return np.array([]), np.array([])

def main():
    """Main execution"""
    print("🚀 ADVANCED GEOPHONE ANALYZER")
    print("FFT + SCTN Feature Analysis")
    print("=" * 50)
    
    analyzer = AdvancedGeophoneAnalyzer()
    
    # Data paths
    data_dir = "project/data"
    chunks_dir = "project/MyCode/chunked_output"
    
    data_files = {
        'car': f"{data_dir}/car.csv",
        'car_nothing': f"{data_dir}/car_nothing.csv",
        'human': f"{data_dir}/human.csv", 
        'human_nothing': f"{data_dir}/human_nothing.csv"
    }
    
    # Extract features
    features_dict = {}
    for signal_type, data_file in data_files.items():
        if os.path.exists(data_file):
            features = analyzer.create_combined_features(data_file, chunks_dir, signal_type)
            features_dict[signal_type] = features
        else:
            features_dict[signal_type] = np.array([])
    
    # Train car classifier
    if features_dict['car'].size > 0 or features_dict['car_nothing'].size > 0:
        car_features = {
            'car': features_dict['car'],
            'car_nothing': features_dict['car_nothing']
        }
        car_results = analyzer.train_classifier(car_features, "CAR")
    
    # Train human classifier
    if features_dict['human'].size > 0 or features_dict['human_nothing'].size > 0:
        human_features = {
            'human': features_dict['human'],
            'human_nothing': features_dict['human_nothing']
        }
        human_results = analyzer.train_classifier(human_features, "HUMAN")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main() 