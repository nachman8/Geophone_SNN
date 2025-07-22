#!/usr/bin/env python3
"""
Ultimate Car Classifier Optimizer
Test if we can maintain/improve 100% car accuracy using advanced techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import pickle
import os
import sys

# Add the directory CONTAINING sctnN to your Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY

class UltimateCarOptimizer:
    """Ultimate optimization system for car classification"""
    
    def __init__(self, chunk_dir="/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"):
        self.chunk_dir = chunk_dir
        self.best_models = []
        self.ensemble_weights = []
        
    def load_chunk_data(self, category, chunk_num):
        """Load chunk data with error handling"""
        chunk_path = os.path.join(self.chunk_dir, category, f"chunk_{chunk_num}", f"chunk_{chunk_num}_data.pkl")
        
        try:
            if os.path.exists(chunk_path):
                with open(chunk_path, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            return None

    def extract_enhanced_features(self, chunk_data):
        """Extract enhanced 32-feature set optimized for car detection"""
        if chunk_data is None:
            return np.zeros(32)
            
        features = []
        
        # ORIGINAL PROVEN FEATURES (16 features)
        if 'spikes_bands_spectrogram' in chunk_data:
            bands = chunk_data['spikes_bands_spectrogram']
            
            if bands.shape[0] >= 8:
                band_energies = [np.sum(bands[i]**2) for i in range(8)]
                total_energy = sum(band_energies) + 1e-8
                
                # Core discriminative ratios
                car_signature = (band_energies[1] + band_energies[2] + band_energies[3]) / total_energy
                human_signature = (band_energies[5] + band_energies[6]) / total_energy
                car_peak_ratio = band_energies[2] / total_energy
                human_peak_ratio = band_energies[5] / total_energy
                
                features.extend([car_signature, human_signature, car_peak_ratio, human_peak_ratio])
                
                # Activity indicators
                car_peak_max = np.max(bands[2])
                human_peak_max = np.max(bands[5])
                car_peak_avg = np.mean(bands[2])
                human_peak_avg = np.mean(bands[5])
                
                features.extend([car_peak_max, human_peak_max, car_peak_avg, human_peak_avg])
            else:
                features.extend([0.0] * 8)
        else:
            features.extend([0.0] * 8)
        
        # Temporal patterns
        if 'max_spikes_spectrogram' in chunk_data:
            max_spikes = chunk_data['max_spikes_spectrogram']
            features.extend([
                np.max(max_spikes),
                np.mean(max_spikes),
                np.std(max_spikes),
                np.sum(max_spikes > np.percentile(max_spikes, 90))
            ])
        else:
            features.extend([0.0] * 4)
        
        # Signal characteristics
        if 'signal' in chunk_data:
            signal = chunk_data['signal']
            features.extend([
                np.std(signal),
                np.max(np.abs(signal)),
                np.mean(np.abs(signal)),
                np.sum(np.abs(signal) > 0.1) / len(signal)
            ])
        else:
            features.extend([0.0] * 4)
        
        # ADVANCED CAR-SPECIFIC FEATURES (16 additional features)
        if 'spikes_bands_spectrogram' in chunk_data and bands.shape[0] >= 8:
            # Car-specific patterns (30-50 Hz dominance)
            car_bands = [1, 2, 3, 4]
            car_energy = sum([np.sum(bands[i]**2) for i in car_bands])
            car_dominance = car_energy / (total_energy + 1e-8)
            
            # Periodicity detection
            car_activity = np.mean(bands[car_bands], axis=0) if len(car_bands) > 0 else np.zeros(bands.shape[1])
            if len(car_activity) > 20:
                fft_car = np.abs(np.fft.fft(car_activity))
                peak_prominence = np.max(fft_car[1:len(fft_car)//2])
                baseline = np.mean(fft_car[1:len(fft_car)//2])
                periodicity_strength = min(peak_prominence / (baseline + 1e-8), 20.0)
                
                regularity = 1.0 / (np.std(car_activity) / (np.mean(car_activity) + 1e-8) + 1e-8)
            else:
                periodicity_strength = regularity = 0
            
            # Low frequency dominance
            low_freq_energy = np.sum(bands[0:4])
            high_freq_energy = np.sum(bands[4:8])
            low_freq_dominance = low_freq_energy / (high_freq_energy + 1e-8)
            
            # Interaction features
            car_human_ratio = car_signature / (human_signature + 1e-8)
            peak_ratio_diff = car_peak_ratio - human_peak_ratio
            
            car_spectral_concentration = car_energy / (np.sum(bands)**2 + 1e-8)
            
            features.extend([
                car_dominance, periodicity_strength, regularity, low_freq_dominance,
                car_human_ratio, peak_ratio_diff, car_spectral_concentration
            ])
        else:
            features.extend([0.0] * 7)
        
        # Advanced temporal analysis
        if 'max_spikes_spectrogram' in chunk_data:
            max_spikes = chunk_data['max_spikes_spectrogram']
            
            # Sustained activity
            if len(max_spikes) > 10:
                sustained_threshold = np.mean(max_spikes) + 0.5 * np.std(max_spikes)
                sustained_activity = np.sum(max_spikes > sustained_threshold) / len(max_spikes)
            else:
                sustained_activity = 0
            
            # Energy distribution
            if len(max_spikes) > 0:
                energy = max_spikes ** 2
                sorted_energy = np.sort(energy)[::-1]
                top_20_percent = int(0.2 * len(sorted_energy))
                energy_distribution = np.sum(sorted_energy[:top_20_percent]) / np.sum(energy) if top_20_percent > 0 else 0
            else:
                energy_distribution = 0
            
            # Consistency measure
            if len(max_spikes) > 10:
                segments = np.array_split(max_spikes, 5)
                segment_means = [np.mean(seg) for seg in segments if len(seg) > 0]
                consistency = 1.0 / (np.std(segment_means) + 1e-8) if len(segment_means) > 1 else 0
            else:
                consistency = 0
            
            # Temporal smoothness
            if len(max_spikes) > 1:
                smoothness = 1.0 / (np.mean(np.abs(np.diff(max_spikes))) + 1e-8)
            else:
                smoothness = 0
            
            # Long-term correlation
            if len(max_spikes) >= 50:
                signal_norm = (max_spikes - np.mean(max_spikes)) / (np.std(max_spikes) + 1e-8)
                n = len(signal_norm)
                autocorr = np.correlate(signal_norm, signal_norm, mode='full')
                autocorr = autocorr[n-1:]
                long_term_corr = np.max(autocorr[10:min(n//2, 100)]) if len(autocorr) > 10 else 0
            else:
                long_term_corr = 0
            
            activity_density = np.sum(max_spikes > np.mean(max_spikes)) / len(max_spikes)
            
            if len(max_spikes) > 0:
                amplitude_stability = 1.0 / (np.std(max_spikes) / (np.mean(max_spikes) + 1e-8) + 1e-8)
            else:
                amplitude_stability = 0
            
            if len(max_spikes) >= 20:
                fft_signal = np.abs(np.fft.fft(max_spikes))
                freq_stability = 1.0 / (np.std(fft_signal) / (np.mean(fft_signal) + 1e-8) + 1e-8)
            else:
                freq_stability = 0
            
            # Peak regularity
            if len(max_spikes) > 10:
                threshold = np.mean(max_spikes) + np.std(max_spikes)
                peaks = max_spikes > threshold
                if np.sum(peaks) >= 2:
                    peak_indices = np.where(peaks)[0]
                    if len(peak_indices) > 1:
                        distances = np.diff(peak_indices)
                        peak_regularity = 1.0 / (np.std(distances) + 1) if len(distances) > 1 else 0
                    else:
                        peak_regularity = 0
                else:
                    peak_regularity = 0
            else:
                peak_regularity = 0
            
            features.extend([
                sustained_activity, energy_distribution, consistency, smoothness,
                long_term_corr, activity_density, amplitude_stability, freq_stability, peak_regularity
            ])
        else:
            features.extend([0.0] * 9)
        
        return np.array(features[:32], dtype=np.float32)

    def create_optimized_car_sctn(self):
        """Create optimized SCTN for car detection"""
        
        class OptimizedCarSCTN:
            def __init__(self, input_size):
                self.input_size = input_size
                self.neuron = None
                self.scaler = None
                
            def _create_neuron(self):
                neuron = create_SCTN()
                neuron.synapses_weights = np.random.normal(0, 0.04, self.input_size).astype(np.float64)
                neuron.threshold_pulse = 20.0
                neuron.activation_function = BINARY
                neuron.theta = 0.0
                neuron.reset_to = 0.0
                neuron.membrane_should_reset = True
                return neuron
            
            def _forward(self, features):
                self.neuron.membrane_potential = 0.0
                self.neuron.index = 0
                activation = np.dot(features, self.neuron.synapses_weights)
                self.neuron.membrane_potential = activation
                output = self.neuron._activation_function_binary()
                return output, activation
            
            def train(self, X, y, epochs=120, lr=0.12):
                self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))
                X_scaled = self.scaler.fit_transform(X)
                self.neuron = self._create_neuron()
                
                for epoch in range(epochs):
                    if epoch <= 40:
                        current_lr = lr * 1.5
                    elif epoch <= 80:
                        current_lr = lr
                    else:
                        current_lr = lr * 0.8
                    
                    indices = np.random.permutation(len(X_scaled))
                    
                    for idx in indices:
                        features = X_scaled[idx]
                        target = y[idx]
                        
                        prediction, activation = self._forward(features)
                        error = target - prediction
                        
                        weight_update = current_lr * error * features
                        self.neuron.synapses_weights += weight_update
                        self.neuron.threshold_pulse += current_lr * error * 0.05
                
                return True
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                predictions = []
                for features in X_scaled:
                    prediction, _ = self._forward(features)
                    predictions.append(prediction)
                return np.array(predictions)
        
        return OptimizedCarSCTN

    def train_ensemble(self, X, y, n_models=7):
        """Train ensemble for car detection"""
        print(f"🎭 Training car ensemble ({n_models} models)...")
        
        models = []
        weights = []
        
        for i in range(n_models):
            print(f"   Training model {i+1}/{n_models}...")
            
            n_samples = len(X)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_bootstrap, y_bootstrap, test_size=0.2, stratify=y_bootstrap, random_state=i*456
            )
            
            model_class = self.create_optimized_car_sctn()
            model = model_class(input_size=X.shape[1])
            
            epochs = 120 + np.random.randint(-15, 16)
            lr = 0.12 + np.random.uniform(-0.02, 0.02)
            
            model.train(X_train, y_train, epochs=epochs, lr=lr)
            
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            models.append(model)
            weights.append(val_accuracy)
            
            print(f"      Validation accuracy: {val_accuracy:.4f}")
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        self.best_models = models
        self.ensemble_weights = weights
        
        return models, weights

    def ensemble_predict(self, X):
        """Ensemble prediction"""
        all_predictions = []
        for model in self.best_models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        weighted_predictions = np.average(all_predictions, axis=0, weights=self.ensemble_weights)
        final_predictions = (weighted_predictions > 0.5).astype(int)
        
        return final_predictions

    def load_car_dataset(self):
        """Load car dataset"""
        print("🔄 Loading car dataset...")
        
        categories = {'car': 28, 'car_nothing': 16}
        all_features = []
        all_labels = []
        
        for category, max_chunks in categories.items():
            features_list = []
            
            for chunk_num in range(max_chunks):
                chunk_data = self.load_chunk_data(category, chunk_num)
                if chunk_data is not None:
                    features = self.extract_enhanced_features(chunk_data)
                    features_list.append(features)
            
            if features_list:
                category_features = np.array(features_list)
                all_features.append(category_features)
                
                labels = np.ones(len(features_list)) if category == 'car' else np.zeros(len(features_list))
                all_labels.append(labels)
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            return X, y
        
        return None, None

    def run_ultimate_car_optimization(self):
        """Run ultimate optimization for car classification"""
        print("🚀 ULTIMATE CAR OPTIMIZATION - CAN WE BEAT 100%?")
        print("=" * 65)
        
        X, y = self.load_car_dataset()
        if X is None:
            print("❌ Failed to load dataset")
            return None
        
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"📊 Labels: {Counter(y)}")
        
        # Feature selection
        print(f"\n1️⃣ FEATURE OPTIMIZATION")
        if X.shape[1] > 20:
            selector = SelectKBest(score_func=f_classif, k=20)
            X_selected = selector.fit_transform(X, y)
            print(f"   Features: {X.shape[1]} → {X_selected.shape[1]}")
        else:
            X_selected = X
            selector = None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.30, stratify=y, random_state=42
        )
        
        # Ensemble training
        print(f"\n2️⃣ ENSEMBLE TRAINING")
        models, weights = self.train_ensemble(X_train, y_train, n_models=7)
        
        # Evaluation
        print(f"\n3️⃣ FINAL EVALUATION")
        y_pred = self.ensemble_predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Results
        baseline = 1.0000
        improvement = accuracy - baseline
        
        print(f"\n🎯 ULTIMATE CAR RESULTS")
        print("=" * 50)
        print(f"Baseline (current): {baseline:.4f} (100.00%)")
        print(f"Ultimate optimized: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Improvement:        {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if accuracy >= 1.0:
            print("🎉 PERFECT ACCURACY MAINTAINED/ACHIEVED!")
        elif accuracy >= 0.95:
            print("🎯 EXCELLENT PERFORMANCE!")
        
        # Detailed analysis
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📋 DETAILED RESULTS:")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Test labels: {Counter(y_test)}")
        print(f"   Predictions: {Counter(y_pred)}")
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n📈 Confusion Matrix:")
            print(f"                 Predicted")
            print(f"Actual    Nothing  Signal")
            print(f"Nothing     {tn:4d}    {fp:4d}")
            print(f"Signal      {fn:4d}    {tp:4d}")
        
        return {
            'accuracy': accuracy,
            'improvement': improvement,
            'confusion_matrix': cm
        }

def main():
    optimizer = UltimateCarOptimizer()
    return optimizer.run_ultimate_car_optimization()

if __name__ == "__main__":
    main() 