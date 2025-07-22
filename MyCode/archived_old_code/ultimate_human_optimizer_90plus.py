#!/usr/bin/env python3
"""
Ultimate Human Classifier Optimizer
Targets 90%+ accuracy for human footstep detection using advanced techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import pickle
import time
import os
import sys

# Add the directory CONTAINING sctnN to your Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY

class UltimateHumanOptimizer:
    """
    Ultimate optimization system targeting 90%+ human classification accuracy
    """
    
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
        """Extract enhanced 32-feature set for maximum discrimination"""
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
        
        # ADVANCED FEATURES (16 additional features)
        if 'spikes_bands_spectrogram' in chunk_data and bands.shape[0] >= 8:
            # Human-specific patterns
            human_bands = [5, 6, 7]
            human_energy = sum([np.sum(bands[i]**2) for i in human_bands])
            human_dominance = human_energy / (total_energy + 1e-8)
            
            # Burst detection
            human_activity = np.mean(bands[human_bands], axis=0) if len(human_bands) > 0 else np.zeros(bands.shape[1])
            if len(human_activity) > 10:
                threshold = np.mean(human_activity) + 1.5 * np.std(human_activity)
                bursts = human_activity > threshold
                burst_count = np.sum(np.diff(np.concatenate([[False], bursts, [False]])) == 1)
                burst_intensity = np.mean(human_activity[bursts]) if np.any(bursts) else 0
            else:
                burst_count = burst_intensity = 0
            
            # Variability measures
            temporal_variability = np.std(human_activity) / (np.mean(human_activity) + 1e-8)
            
            # Interaction features
            human_car_ratio = human_signature / (car_signature + 1e-8)
            peak_ratio_diff = human_peak_ratio - car_peak_ratio
            
            # Spectral features
            high_freq_energy = np.sum(bands[6:8]) if bands.shape[0] > 6 else 0
            low_freq_energy = np.sum(bands[0:3])
            freq_balance = high_freq_energy / (low_freq_energy + 1e-8)
            
            features.extend([
                human_dominance, burst_count, burst_intensity, temporal_variability,
                human_car_ratio, peak_ratio_diff, freq_balance
            ])
        else:
            features.extend([0.0] * 7)
        
        # Advanced temporal analysis
        if 'max_spikes_spectrogram' in chunk_data:
            max_spikes = chunk_data['max_spikes_spectrogram']
            
            # Event clustering
            if len(max_spikes) > 10:
                threshold = np.mean(max_spikes) + np.std(max_spikes)
                peaks = max_spikes > threshold
                if np.sum(peaks) >= 2:
                    peak_indices = np.where(peaks)[0]
                    distances = np.diff(peak_indices)
                    clustering = 1.0 / (np.var(distances) + 1) if len(distances) > 1 else 0
                else:
                    clustering = 0
            else:
                clustering = 0
            
            # Energy concentration
            energy = max_spikes ** 2
            sorted_energy = np.sort(energy)[::-1]
            top_10_percent = int(0.1 * len(sorted_energy))
            concentration = np.sum(sorted_energy[:top_10_percent]) / np.sum(energy) if top_10_percent > 0 else 0
            
            # Activity persistence
            if len(max_spikes) > 0:
                threshold = np.mean(max_spikes)
                above_threshold = max_spikes > threshold
                runs = []
                current_run = 0
                for val in above_threshold.flatten():
                    if val:
                        current_run += 1
                    else:
                        if current_run > 0:
                            runs.append(current_run)
                        current_run = 0
                if current_run > 0:
                    runs.append(current_run)
                persistence = np.mean(runs) if runs else 0
            else:
                persistence = 0
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(max_spikes - np.mean(max_spikes))) != 0)
            zcr = zero_crossings / len(max_spikes) if len(max_spikes) > 1 else 0
            
            # Spectral entropy
            if len(max_spikes) >= 10:
                fft_signal = np.abs(np.fft.fft(max_spikes))
                psd = fft_signal ** 2
                psd_norm = psd / (np.sum(psd) + 1e-8)
                entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
            else:
                entropy = 0
            
            # Autocorrelation peak
            if len(max_spikes) >= 20:
                signal_norm = (max_spikes - np.mean(max_spikes)) / (np.std(max_spikes) + 1e-8)
                n = len(signal_norm)
                autocorr = np.correlate(signal_norm, signal_norm, mode='full')
                autocorr = autocorr[n-1:]
                peak_idx = np.argmax(autocorr[1:n//4]) + 1 if len(autocorr) > 1 else 0
                autocorr_peak = peak_idx / len(signal_norm)
            else:
                autocorr_peak = 0
            
            # Periodicity
            if len(max_spikes) >= 20:
                fft_signal = np.abs(np.fft.fft(max_spikes))
                peak_prominence = np.max(fft_signal[1:len(fft_signal)//2])
                baseline = np.mean(fft_signal[1:len(fft_signal)//2])
                periodicity = min(peak_prominence / (baseline + 1e-8), 10.0)
            else:
                periodicity = 0
            
            # Activity ratio (percentage of time above mean)
            activity_ratio = np.sum(max_spikes > np.mean(max_spikes)) / len(max_spikes)
            
            # Peak sharpness
            peaks_above_mean = max_spikes > np.mean(max_spikes)
            if np.any(peaks_above_mean):
                peak_values = max_spikes[peaks_above_mean]
                sharpness = np.std(peak_values) / (np.mean(peak_values) + 1e-8)
            else:
                sharpness = 0
            
            features.extend([
                clustering, concentration, persistence, zcr, entropy,
                autocorr_peak, periodicity, activity_ratio, sharpness
            ])
        else:
            features.extend([0.0] * 9)
        
        return np.array(features[:32], dtype=np.float32)

    def create_super_augmented_dataset(self, X, y, augmentation_factor=5):
        """Create comprehensive augmented dataset"""
        print(f"🔬 Creating super-augmented dataset (factor={augmentation_factor})...")
        
        X_augmented = list(X)
        y_augmented = list(y)
        
        for aug_type in range(augmentation_factor):
            for i in range(len(X)):
                if aug_type == 0:
                    # Gaussian noise (class-specific)
                    noise_level = 0.02 if y[i] == 1 else 0.04
                    noise = np.random.normal(0, noise_level, X[i].shape)
                    X_noisy = X[i] + noise
                    
                elif aug_type == 1:
                    # Feature dropout
                    X_dropout = X[i].copy()
                    mask = np.random.random(X[i].shape) > 0.08
                    X_dropout = X_dropout * mask
                    X_noisy = X_dropout
                    
                elif aug_type == 2:
                    # Scaling
                    scale = np.random.uniform(0.92, 1.08)
                    X_noisy = X[i] * scale
                    
                elif aug_type == 3:
                    # Feature shuffling (slight)
                    X_noisy = X[i].copy()
                    n_shuffle = int(0.1 * len(X[i]))
                    shuffle_indices = np.random.choice(len(X[i]), n_shuffle, replace=False)
                    X_noisy[shuffle_indices] = np.random.permutation(X_noisy[shuffle_indices])
                    
                elif aug_type == 4:
                    # Mixup with same class
                    same_class_indices = np.where(y == y[i])[0]
                    if len(same_class_indices) > 1:
                        other_idx = np.random.choice([idx for idx in same_class_indices if idx != i])
                        alpha = np.random.beta(0.4, 0.4)
                        X_noisy = alpha * X[i] + (1 - alpha) * X[other_idx]
                    else:
                        X_noisy = X[i] + np.random.normal(0, 0.01, X[i].shape)
                
                X_noisy = np.clip(X_noisy, -10, 10)
                X_augmented.append(X_noisy)
                y_augmented.append(y[i])
        
        print(f"   📊 Dataset: {len(X)} → {len(X_augmented)} samples")
        return np.array(X_augmented), np.array(y_augmented)

    def create_ultra_optimized_sctn(self):
        """Create ultra-optimized SCTN for human detection"""
        
        class UltraOptimizedSCTN:
            def __init__(self, input_size):
                self.input_size = input_size
                self.neuron = None
                self.scaler = None
                self.is_trained = False
                
            def _create_neuron(self):
                neuron = create_SCTN()
                
                # Ultra-optimized for human footsteps
                neuron.synapses_weights = np.random.normal(0, 0.025, self.input_size).astype(np.float64)
                neuron.threshold_pulse = 12.0  # Highly sensitive
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
            
            def train(self, X, y, epochs=180, lr=0.18):
                # Advanced scaling specifically for human features
                self.scaler = MinMaxScaler(feature_range=(0.05, 0.95))
                X_scaled = self.scaler.fit_transform(X)
                
                self.neuron = self._create_neuron()
                
                # Ultra-optimized training
                for epoch in range(epochs):
                    # Dynamic learning schedule
                    if epoch <= 50:
                        current_lr = lr * 2.0  # Aggressive start
                    elif epoch <= 120:
                        current_lr = lr * 1.3  # Moderate
                    else:
                        current_lr = lr * 0.7  # Fine-tuning
                    
                    # Randomize order each epoch
                    indices = np.random.permutation(len(X_scaled))
                    
                    for idx in indices:
                        features = X_scaled[idx]
                        target = y[idx]
                        
                        prediction, activation = self._forward(features)
                        error = target - prediction
                        
                        # Enhanced weight update with momentum
                        weight_update = current_lr * error * features
                        
                        if hasattr(self, 'momentum'):
                            weight_update += 0.15 * self.momentum
                        
                        self.neuron.synapses_weights += weight_update
                        self.momentum = weight_update
                        
                        # Threshold adaptation
                        self.neuron.threshold_pulse += current_lr * error * 0.025
                
                self.is_trained = True
                return True
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                predictions = []
                for features in X_scaled:
                    prediction, _ = self._forward(features)
                    predictions.append(prediction)
                return np.array(predictions)
        
        return UltraOptimizedSCTN

    def train_ensemble(self, X, y, n_models=8):
        """Train ensemble of ultra-optimized models"""
        print(f"🎭 Training ensemble of {n_models} ultra-optimized models...")
        
        models = []
        weights = []
        
        for i in range(n_models):
            print(f"   Training model {i+1}/{n_models}...")
            
            # Bootstrap sampling
            n_samples = len(X)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_bootstrap, y_bootstrap, test_size=0.18, stratify=y_bootstrap, random_state=i*123
            )
            
            # Create model
            model_class = self.create_ultra_optimized_sctn()
            model = model_class(input_size=X.shape[1])
            
            # Vary parameters slightly
            epochs = 180 + np.random.randint(-25, 26)
            lr = 0.18 + np.random.uniform(-0.04, 0.04)
            
            model.train(X_train, y_train, epochs=epochs, lr=lr)
            
            # Validate
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            models.append(model)
            weights.append(val_accuracy)
            
            print(f"      Validation accuracy: {val_accuracy:.4f}")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        self.best_models = models
        self.ensemble_weights = weights
        
        return models, weights

    def ensemble_predict(self, X):
        """Ensemble prediction with weighted voting"""
        if not self.best_models:
            raise ValueError("Ensemble not trained")
        
        all_predictions = []
        for model in self.best_models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        weighted_predictions = np.average(all_predictions, axis=0, weights=self.ensemble_weights)
        final_predictions = (weighted_predictions > 0.5).astype(int)
        
        return final_predictions

    def load_human_dataset(self):
        """Load human dataset"""
        print("🔄 Loading human dataset...")
        
        categories = {'human': 47, 'human_nothing': 33}
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
                
                labels = np.ones(len(features_list)) if category == 'human' else np.zeros(len(features_list))
                all_labels.append(labels)
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            return X, y
        
        return None, None

    def run_ultimate_optimization(self):
        """Run ultimate optimization targeting 90%+"""
        print("🚀 ULTIMATE HUMAN OPTIMIZATION FOR 90%+ ACCURACY")
        print("=" * 65)
        
        # Load data
        X, y = self.load_human_dataset()
        if X is None:
            print("❌ Failed to load dataset")
            return None
        
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"📊 Labels: {Counter(y)}")
        
        # Feature selection
        print(f"\n1️⃣ FEATURE OPTIMIZATION")
        if X.shape[1] > 24:
            selector = SelectKBest(score_func=f_classif, k=24)
            X_selected = selector.fit_transform(X, y)
            print(f"   Features: {X.shape[1]} → {X_selected.shape[1]}")
        else:
            X_selected = X
            selector = None
        
        # Train/test split
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X_selected, y, test_size=0.32, stratify=y, random_state=42
        )
        
        # Data augmentation
        print(f"\n2️⃣ SUPER AUGMENTATION")
        X_train_aug, y_train_aug = self.create_super_augmented_dataset(
            X_train_orig, y_train_orig, augmentation_factor=6
        )
        
        # Ensemble training
        print(f"\n3️⃣ ENSEMBLE TRAINING")
        models, weights = self.train_ensemble(X_train_aug, y_train_aug, n_models=10)
        
        # Evaluation
        print(f"\n4️⃣ FINAL EVALUATION")
        y_pred = self.ensemble_predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Results
        baseline = 0.8750
        improvement = accuracy - baseline
        
        print(f"\n🎯 ULTIMATE RESULTS")
        print("=" * 40)
        print(f"Baseline (current): {baseline:.4f} (87.50%)")
        print(f"Ultimate optimized: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Improvement:        {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if accuracy >= 0.90:
            print("🎉 TARGET ACHIEVED! 90%+ ACCURACY!")
        elif accuracy > baseline:
            print("📈 SIGNIFICANT IMPROVEMENT!")
        
        # Save results
        results = {
            'ensemble_models': self.best_models,
            'ensemble_weights': self.ensemble_weights,
            'feature_selector': selector,
            'accuracy': accuracy,
            'improvement': improvement,
            'test_data': (X_test, y_test)
        }
        
        with open('ultimate_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Saved: ultimate_results.pkl")
        return results

def main():
    optimizer = UltimateHumanOptimizer()
    return optimizer.run_ultimate_optimization()

if __name__ == "__main__":
    main() 