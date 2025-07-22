#!/usr/bin/env python3
"""
Final Comprehensive Geophone Analyzer
Combines all the best approaches from the old notebooks
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import welch, find_peaks, correlate
from scipy.stats import skew, kurtosis

class FinalAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def analyze_raw_data(self, csv_path):
        print(f"📄 Analyzing: {os.path.basename(csv_path)}")
        
        data = pd.read_csv(csv_path)
        time_series = data['time_s'].values
        amplitude = data['amplitude'].values
        
        dt = np.mean(np.diff(time_series))
        fs = 1.0 / dt
        
        segment_duration = 30
        segment_samples = int(segment_duration * fs)
        n_segments = len(amplitude) // segment_samples
        
        features_list = []
        for i in range(n_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            segment = amplitude[start_idx:end_idx]
            
            features = self._extract_fft_features(segment, fs)
            features_list.append(features)
        
        print(f"   ✅ {len(features_list)} segments")
        return np.array(features_list)
    
    def _extract_fft_features(self, segment, fs):
        features = []
        
        freqs, psd = welch(segment, fs, nperseg=min(len(segment)//4, 512))
        
        # Band powers
        bands = [(20,30), (30,34), (34,40), (40,48), (48,60), (60,70), (70,85), (85,100)]
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[mask]) if np.any(mask) else 0
            features.append(band_power)
        
        # Cross-band ratios
        total_power = np.sum(psd)
        car_power = np.sum(psd[(freqs >= 30) & (freqs <= 48)])
        human_power = np.sum(psd[(freqs >= 60) & (freqs <= 85)])
        
        features.extend([
            car_power / (total_power + 1e-10),
            human_power / (total_power + 1e-10),
            car_power / (human_power + 1e-10)
        ])
        
        return features
    
    def analyze_chunks(self, chunks_dir, signal_type):
        print(f"📦 Analyzing {signal_type} chunks")
        
        index_file = os.path.join(chunks_dir, signal_type, 'chunk_index.pkl')
        if not os.path.exists(index_file):
            print(f"   ❌ No chunks found")
            return np.array([])
        
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        chunks = []
        for chunk_file in chunk_index['chunk_files']:
            if os.path.exists(chunk_file):
                with open(chunk_file, 'rb') as f:
                    chunk = pickle.load(f)
                chunks.append(chunk)
        
        features_list = []
        for chunk in chunks:
            features = self._extract_chunk_features(chunk)
            features_list.append(features)
        
        print(f"   ✅ {len(chunks)} chunks processed")
        return np.array(features_list)
    
    def _extract_chunk_features(self, chunk):
        features = []
        spikegram = chunk['spikes_bands_spectrogram']
        
        # Band statistics
        for band_idx in range(8):
            band_data = spikegram[band_idx]
            features.extend([
                np.mean(band_data),
                np.std(band_data),
                np.max(band_data)
            ])
        
        # Car pattern (bands 1-3)
        car_bands = spikegram[1:4]
        car_pattern = np.mean(car_bands, axis=0)
        car_periodicity = self._detect_periodicity(car_pattern)
        
        # Human pattern (bands 5-7)
        human_bands = spikegram[5:8]
        human_pattern = np.mean(human_bands, axis=0)
        human_burstiness = self._detect_burstiness(human_pattern)
        
        features.extend([
            car_periodicity,
            human_burstiness,
            np.mean(car_bands),
            np.mean(human_bands)
        ])
        
        return features
    
    def _detect_periodicity(self, signal):
        if len(signal) < 20:
            return 0
        
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        autocorr = correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 10:
            peaks, _ = find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
            return len(peaks) / len(autocorr) * 100
        return 0
    
    def _detect_burstiness(self, signal):
        if len(signal) == 0:
            return 0
        
        threshold = np.mean(signal) + 2 * np.std(signal)
        bursts = signal > threshold
        
        if np.sum(bursts) == 0:
            return 0
        
        burst_changes = np.diff(bursts.astype(int))
        burst_starts = np.where(burst_changes == 1)[0]
        
        if len(burst_starts) < 2:
            return 0
        
        intervals = np.diff(burst_starts)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        return (cv - 1) / (cv + 1) if cv > 0 else 0
    
    def train_classifier(self, X, y, task_name):
        print(f"\n🎯 {task_name}")
        print("=" * 40)
        
        if len(X) == 0:
            print("❌ No data")
            return None
        
        print(f"📊 {len(X)} samples, {X.shape[1]} features")
        print(f"📈 Classes: {np.bincount(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        best_acc = 0
        best_name = ""
        
        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, y_train)
            test_acc = clf.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            
            print(f"{name:<15} Test: {test_acc:.1%}  CV: {cv_scores.mean():.1%}")
            
            if cv_scores.mean() > best_acc:
                best_acc = cv_scores.mean()
                best_name = name
                best_clf = clf
        
        y_pred = best_clf.predict(X_test_scaled)
        print(f"\n🏆 BEST: {best_name} ({best_clf.score(X_test_scaled, y_test):.1%})")
        print(classification_report(y_test, y_pred, target_names=['Signal', 'Nothing']))
        
        return best_clf

def main():
    print("🚀 FINAL COMPREHENSIVE ANALYZER")
    print("=" * 50)
    
    analyzer = FinalAnalyzer()
    
    base_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo"
    data_dir = f"{base_dir}/project/data"
    chunks_dir = f"{base_dir}/project/MyCode/chunked_output"
    
    all_features = {}
    
    for signal_type in ['car', 'car_nothing', 'human', 'human_nothing']:
        print(f"\n🔍 {signal_type.upper()}")
        
        # Raw data
        csv_path = f"{data_dir}/{signal_type}.csv"
        if os.path.exists(csv_path):
            fft_features = analyzer.analyze_raw_data(csv_path)
        else:
            fft_features = np.array([])
        
        # Chunks
        sctn_features = analyzer.analyze_chunks(chunks_dir, signal_type)
        
        # Combine
        if fft_features.size > 0 and sctn_features.size > 0:
            min_samples = min(len(fft_features), len(sctn_features))
            if len(fft_features) > min_samples:
                indices = np.linspace(0, len(fft_features)-1, min_samples).astype(int)
                fft_features = fft_features[indices]
            if len(sctn_features) > min_samples:
                indices = np.linspace(0, len(sctn_features)-1, min_samples).astype(int)
                sctn_features = sctn_features[indices]
            combined = np.hstack([fft_features, sctn_features])
        elif fft_features.size > 0:
            combined = fft_features
        elif sctn_features.size > 0:
            combined = sctn_features
        else:
            combined = np.array([])
        
        all_features[signal_type] = combined
        print(f"   🔗 Final features: {combined.shape}")
    
    # Train car classifier
    if all_features['car'].size > 0 or all_features['car_nothing'].size > 0:
        X_car, y_car = [], []
        if all_features['car'].size > 0:
            X_car.append(all_features['car'])
            y_car.append(np.zeros(len(all_features['car'])))
        if all_features['car_nothing'].size > 0:
            X_car.append(all_features['car_nothing'])
            y_car.append(np.ones(len(all_features['car_nothing'])))
        
        if X_car:
            X_car = np.vstack(X_car)
            y_car = np.concatenate(y_car).astype(int)
            analyzer.train_classifier(X_car, y_car, "CAR DETECTION")
    
    # Train human classifier
    if all_features['human'].size > 0 or all_features['human_nothing'].size > 0:
        X_human, y_human = [], []
        if all_features['human'].size > 0:
            X_human.append(all_features['human'])
            y_human.append(np.zeros(len(all_features['human'])))
        if all_features['human_nothing'].size > 0:
            X_human.append(all_features['human_nothing'])
            y_human.append(np.ones(len(all_features['human_nothing'])))
        
        if X_human:
            X_human = np.vstack(X_human)
            y_human = np.concatenate(y_human).astype(int)
            analyzer.train_classifier(X_human, y_human, "HUMAN DETECTION")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main() 