#!/usr/bin/env python3
"""
Comprehensive Geophone Analyzer
Implements best practices from EEG study and resonator analysis
Raw data FFT + Chunks SCTN + Advanced pattern detection
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import welch, find_peaks, correlate
from scipy.stats import skew, kurtosis

class ComprehensiveAnalyzer:
    """Comprehensive geophone analysis system"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Frequency bands based on spikegram analysis
        self.frequency_bands = {
            'LOW_FREQ': (20, 30),      # Band 0
            'CAR_APPROACH': (30, 34),  # Band 1  
            'CAR_PEAK': (34, 40),      # Band 2
            'CAR_TAIL': (40, 48),      # Band 3
            'MID_GAP': (48, 60),       # Band 4
            'HUMAN_PEAK': (60, 70),    # Band 5
            'HUMAN_TAIL': (70, 85),    # Band 6  
            'HIGH_FREQ': (85, 100)     # Band 7
        }
        
    def analyze_raw_data(self, csv_path, segment_duration=30):
        """Analyze raw CSV data with FFT features"""
        print(f"📄 Analyzing raw data: {os.path.basename(csv_path)}")
        
        try:
            # Load CSV data
            data = pd.read_csv(csv_path)
            time_series = data['time_s'].values
            amplitude = data['amplitude'].values
            
            # Calculate sampling parameters
            dt = np.mean(np.diff(time_series))
            fs = 1.0 / dt
            
            print(f"   📊 {len(amplitude)} samples, {time_series[-1]:.1f}s, fs={fs:.1f}Hz")
            
            # Extract segments
            segment_samples = int(segment_duration * fs)
            n_segments = len(amplitude) // segment_samples
            
            features_list = []
            for i in range(n_segments):
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                segment = amplitude[start_idx:end_idx]
                
                # Extract FFT features for this segment
                features = self._extract_fft_features(segment, fs)
                features_list.append(features)
            
            print(f"   ✅ {len(features_list)} FFT segments, {len(features_list[0])} features each")
            return np.array(features_list)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return np.array([])
    
    def _extract_fft_features(self, segment, fs):
        """Extract comprehensive FFT features from segment"""
        features = []
        
        # Power spectral density using Welch's method
        freqs, psd = welch(segment, fs, nperseg=min(len(segment)//4, 512))
        
        # Band power analysis
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.sum(psd[band_mask])
                band_peak_freq = freqs[band_mask][np.argmax(psd[band_mask])]
                band_peak_power = np.max(psd[band_mask])
                band_mean_power = np.mean(psd[band_mask])
                
                features.extend([
                    band_power,
                    band_peak_freq, 
                    band_peak_power,
                    band_mean_power
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Cross-band ratios (key discriminative features)
        total_power = np.sum(psd)
        car_power = np.sum(psd[(freqs >= 30) & (freqs <= 48)])  # Car bands
        human_power = np.sum(psd[(freqs >= 60) & (freqs <= 85)])  # Human bands
        low_power = np.sum(psd[(freqs >= 20) & (freqs <= 30)])  # Background
        
        features.extend([
            car_power / (total_power + 1e-10),      # Car dominance
            human_power / (total_power + 1e-10),    # Human dominance  
            car_power / (human_power + 1e-10),      # Car vs human
            (car_power + human_power) / (low_power + 1e-10),  # Signal vs noise
            total_power,                            # Total energy
            freqs[np.argmax(psd)],                  # Peak frequency
            np.max(psd)                             # Peak power
        ])
        
        return features
    
    def analyze_chunks(self, chunks_dir, signal_type):
        """Analyze saved chunks with SCTN features"""
        print(f"📦 Analyzing {signal_type} chunks")
        
        try:
            # Load chunk index
            index_file = os.path.join(chunks_dir, signal_type, 'chunk_index.pkl')
            if not os.path.exists(index_file):
                print(f"   ❌ Chunk index not found: {index_file}")
                return np.array([])
            
            with open(index_file, 'rb') as f:
                chunk_index = pickle.load(f)
            
            # Load chunks
            chunks = []
            for chunk_file in chunk_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk = pickle.load(f)
                    chunks.append(chunk)
            
            print(f"   📦 {len(chunks)} chunks loaded")
            
            # Extract features from each chunk
            features_list = []
            for chunk in chunks:
                features = self._extract_chunk_features(chunk)
                features_list.append(features)
            
            print(f"   ✅ {len(features_list[0])} SCTN features per chunk")
            return np.array(features_list)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return np.array([])
    
    def _extract_chunk_features(self, chunk):
        """Extract advanced features from chunk using pattern analysis"""
        features = []
        
        # Get spikegram data (8 bands x time_bins)
        spikegram = chunk['spikes_bands_spectrogram']
        
        # Basic band statistics
        for band_idx in range(8):
            band_data = spikegram[band_idx]
            
            features.extend([
                np.mean(band_data),                    # Mean activity
                np.std(band_data),                     # Activity variation
                np.max(band_data),                     # Peak activity
                skew(band_data),                       # Activity skewness
                kurtosis(band_data),                   # Activity kurtosis
                np.sum(band_data > 0) / len(band_data) # Activity ratio
            ])
        
        # Car-specific pattern analysis (bands 1-3: 30-48 Hz)
        car_bands = spikegram[1:4]
        car_pattern = np.mean(car_bands, axis=0)
        
        # Car periodicity (key characteristic)
        car_periodicity = self._detect_periodicity(car_pattern)
        
        # Car consistency across bands
        car_consistency = self._calculate_band_consistency(car_bands)
        
        # Human-specific pattern analysis (bands 5-7: 60-85 Hz)
        human_bands = spikegram[5:8]
        human_pattern = np.mean(human_bands, axis=0)
        
        # Human burstiness (key characteristic)
        human_burstiness = self._detect_burstiness(human_pattern)
        
        # Human sparsity
        human_sparsity = self._calculate_sparsity(human_pattern)
        
        # Cross-pattern features
        car_strength = np.mean(car_bands)
        human_strength = np.mean(human_bands)
        pattern_separation = abs(car_strength - human_strength)
        
        features.extend([
            car_periodicity,
            car_consistency, 
            human_burstiness,
            human_sparsity,
            car_strength,
            human_strength,
            pattern_separation,
            car_strength / (human_strength + 1e-10)
        ])
        
        # Temporal dynamics
        total_activity = np.sum(spikegram, axis=0)
        activity_peaks = find_peaks(total_activity, height=np.mean(total_activity))[0]
        
        features.extend([
            len(activity_peaks),                       # Number of activity peaks
            np.std(total_activity),                    # Activity variation
            np.max(total_activity)                     # Peak total activity
        ])
        
        return features
    
    def _detect_periodicity(self, signal):
        """Detect periodicity using autocorrelation (car signature)"""
        if len(signal) < 20:
            return 0
        
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Autocorrelation
        autocorr = correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find periodic peaks
        if len(autocorr) > 10:
            threshold = 0.1 * np.max(autocorr)
            peaks, _ = find_peaks(autocorr[1:], height=threshold, distance=10)
            return len(peaks) / len(autocorr) * 100
        
        return 0
    
    def _calculate_band_consistency(self, multi_band_data):
        """Calculate consistency across frequency bands"""
        if len(multi_band_data) < 2:
            return 0
        
        correlations = []
        for i in range(len(multi_band_data)):
            for j in range(i+1, len(multi_band_data)):
                corr = np.corrcoef(multi_band_data[i], multi_band_data[j])[0,1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0
    
    def _detect_burstiness(self, signal):
        """Detect burstiness (human footstep signature)"""
        if len(signal) == 0:
            return 0
        
        # Find bursts
        threshold = np.mean(signal) + 2 * np.std(signal)
        bursts = signal > threshold
        
        if np.sum(bursts) == 0:
            return 0
        
        # Burst timing analysis
        burst_changes = np.diff(bursts.astype(int))
        burst_starts = np.where(burst_changes == 1)[0]
        
        if len(burst_starts) < 2:
            return 0
        
        # Inter-burst intervals
        intervals = np.diff(burst_starts)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        
        # Burstiness index
        return (cv - 1) / (cv + 1) if cv > 0 else 0
    
    def _calculate_sparsity(self, signal):
        """Calculate signal sparsity"""
        if len(signal) == 0:
            return 0
        
        threshold = np.mean(signal) + np.std(signal)
        active_ratio = np.sum(signal > threshold) / len(signal)
        return 1.0 - active_ratio  # Sparsity = 1 - activity
    
    def train_classifier(self, X, y, task_name):
        """Train comprehensive classifier"""
        print(f"\n🎯 TRAINING {task_name} CLASSIFIER")
        print("=" * 50)
        
        if len(X) == 0:
            print("❌ No training data")
            return None
        
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"📈 Classes: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Multiple classifiers (EEG study approach)
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f"\n📋 CLASSIFIER COMPARISON:")
        results = {}
        
        for name, clf in classifiers.items():
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
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
        
        # Detailed evaluation
        y_pred = best_clf.predict(X_test_scaled)
        print(f"\n🏆 BEST: {best_name} ({results[best_name]['test_accuracy']:.1%})")
        print(classification_report(y_test, y_pred, target_names=['Signal', 'Nothing']))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        return {
            'classifier': best_clf,
            'scaler': self.scaler,
            'accuracy': results[best_name]['test_accuracy'],
            'results': results
        }

def main():
    """Main analysis execution"""
    print("🚀 COMPREHENSIVE GEOPHONE ANALYZER")
    print("Advanced Pattern Detection: FFT + SCTN + Footprint Analysis")
    print("=" * 70)
    
    analyzer = ComprehensiveAnalyzer()
    
    # Data paths
    base_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo"
    data_dir = f"{base_dir}/project/data"
    chunks_dir = f"{base_dir}/project/MyCode/chunked_output"
    
    print(f"📂 Data directory: {data_dir}")
    print(f"📦 Chunks directory: {chunks_dir}")
    
    # Analyze each signal type
    signal_types = ['car', 'car_nothing', 'human', 'human_nothing']
    all_features = {}
    
    for signal_type in signal_types:
        print(f"\n🔍 PROCESSING {signal_type.upper()}")
        print("-" * 40)
        
        # Raw data analysis
        csv_path = f"{data_dir}/{signal_type}.csv"
        if os.path.exists(csv_path):
            fft_features = analyzer.analyze_raw_data(csv_path)
        else:
            print(f"❌ CSV not found: {csv_path}")
            fft_features = np.array([])
        
        # Chunk analysis
        sctn_features = analyzer.analyze_chunks(chunks_dir, signal_type)
        
        # Combine features
        if fft_features.size > 0 and sctn_features.size > 0:
            # Align feature counts
            min_samples = min(len(fft_features), len(sctn_features))
            
            if len(fft_features) > min_samples:
                indices = np.linspace(0, len(fft_features)-1, min_samples).astype(int)
                fft_features = fft_features[indices]
            
            if len(sctn_features) > min_samples:
                indices = np.linspace(0, len(sctn_features)-1, min_samples).astype(int)
                sctn_features = sctn_features[indices]
            
            combined_features = np.hstack([fft_features, sctn_features])
            print(f"🔗 Combined features: {combined_features.shape}")
            
        elif fft_features.size > 0:
            combined_features = fft_features
            print(f"⚠️ FFT only: {combined_features.shape}")
            
        elif sctn_features.size > 0:
            combined_features = sctn_features
            print(f"⚠️ SCTN only: {combined_features.shape}")
            
        else:
            combined_features = np.array([])
            print(f"❌ No features extracted")
        
        all_features[signal_type] = combined_features
    
    # Train classifiers
    
    # Car vs Car_Nothing
    if all_features['car'].size > 0 or all_features['car_nothing'].size > 0:
        X_car = []
        y_car = []
        
        if all_features['car'].size > 0:
            X_car.append(all_features['car'])
            y_car.append(np.zeros(len(all_features['car'])))  # 0 = signal
        
        if all_features['car_nothing'].size > 0:
            X_car.append(all_features['car_nothing'])
            y_car.append(np.ones(len(all_features['car_nothing'])))  # 1 = nothing
        
        if X_car:
            X_car = np.vstack(X_car)
            y_car = np.concatenate(y_car).astype(int)
            car_results = analyzer.train_classifier(X_car, y_car, "CAR DETECTION")
    
    # Human vs Human_Nothing
    if all_features['human'].size > 0 or all_features['human_nothing'].size > 0:
        X_human = []
        y_human = []
        
        if all_features['human'].size > 0:
            X_human.append(all_features['human'])
            y_human.append(np.zeros(len(all_features['human'])))  # 0 = signal
        
        if all_features['human_nothing'].size > 0:
            X_human.append(all_features['human_nothing'])
            y_human.append(np.ones(len(all_features['human_nothing'])))  # 1 = nothing
        
        if X_human:
            X_human = np.vstack(X_human)
            y_human = np.concatenate(y_human).astype(int)
            human_results = analyzer.train_classifier(X_human, y_human, "HUMAN DETECTION")
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("Advanced footprint detection system trained successfully.")

if __name__ == "__main__":
    main() 