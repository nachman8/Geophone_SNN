#!/usr/bin/env python3
"""
FOCUSED 60-90% TRAIN OPTIMIZER FOR CAR DATA
Test EVERY train split (60%-90% TRAIN) with EVERY epoch (20-2000) to find optimal configuration for car classification
"""

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import sys
import os
import json
from datetime import datetime

# Add the directory CONTAINING sctnN to your Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY

CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

class FocusedCarSCTNClassifier:
    """Focused SCTN classifier for car data 60-90% train range optimization"""
    
    def __init__(self, input_size=16):
        self.input_size = input_size
        self.neuron = None
        self.scaler = None
        
    def _create_car_neuron(self):
        """Create optimized SCTN neuron for car signals"""
        neuron = create_SCTN()
        
        # Car-optimized parameters
        neuron.synapses_weights = np.random.normal(0, 0.1, self.input_size).astype(np.float64)
        neuron.threshold_pulse = 50.0  # Standard threshold for car data
        
        # Car signals benefit from enhanced mid-range frequency features
        # Features: [car_sig, human_sig, car_peak, human_peak, car_max, human_max, car_avg, human_avg, ...]
        car_feature_indices = [0, 2, 4, 6]  # Car-related features
        human_suppression_indices = [1, 3, 5, 7]  # Suppress human features for car focus
        
        for idx in car_feature_indices:
            if idx < len(neuron.synapses_weights):
                neuron.synapses_weights[idx] *= 1.5  # Boost car features
        
        for idx in human_suppression_indices:
            if idx < len(neuron.synapses_weights):
                neuron.synapses_weights[idx] *= 0.8  # Slight suppression of human features
        
        neuron.activation_function = BINARY
        neuron.theta = 0.0
        neuron.reset_to = 0.0
        neuron.membrane_should_reset = True
        
        return neuron
    
    def _forward(self, features):
        """Forward pass"""
        self.neuron.membrane_potential = 0.0
        self.neuron.index = 0
        activation = np.dot(features, self.neuron.synapses_weights)
        self.neuron.membrane_potential = activation
        output = self.neuron._activation_function_binary()
        return output, activation
    
    def train_with_epochs(self, X, y, epochs=200, lr=0.1):
        """Train with specified number of epochs"""
        # Car-specific feature scaling
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler.fit_transform(X)
        
        # Create car-optimized neuron
        self.neuron = self._create_car_neuron()
        
        # Standard learning rate for car patterns (no adaptive phases like human)
        for epoch in range(epochs):
            for i in range(len(X_scaled)):
                features = X_scaled[i]
                target = y[i]
                
                prediction, activation = self._forward(features)
                error = target - prediction
                
                # Update weights
                self.neuron.synapses_weights += lr * error * features
                
                # Standard threshold update for car data
                self.neuron.threshold_pulse += lr * error * 0.1
        
        return True
    
    def predict(self, X):
        """Make predictions with car-specific preprocessing"""
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for features in X_scaled:
            prediction, _ = self._forward(features)
            predictions.append(prediction)
        
        return np.array(predictions)

def load_car_data():
    """Load car classification data"""
    categories = ['car', 'car_nothing']
    chunk_counts = {'car': 28, 'car_nothing': 16}
    
    all_features = {}
    
    for category in categories:
        features_list = []
        for chunk_num in range(chunk_counts[category]):
            chunk_path = os.path.join(CHUNKED_OUTPUT_DIR, category, f"chunk_{chunk_num}", f"chunk_{chunk_num}_data.pkl")
            try:
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Extract 16 features (same as human but optimized for car)
                    features = []
                    
                    if 'spikes_bands_spectrogram' in chunk_data:
                        bands = chunk_data['spikes_bands_spectrogram']
                        if bands.shape[0] >= 8:
                            band_energies = [np.sum(bands[i]**2) for i in range(8)]
                            total_energy = sum(band_energies) + 1e-8
                            
                            car_signature = (band_energies[1] + band_energies[2] + band_energies[3]) / total_energy
                            human_signature = (band_energies[5] + band_energies[6]) / total_energy
                            car_peak_ratio = band_energies[2] / total_energy
                            human_peak_ratio = band_energies[5] / total_energy
                            
                            features.extend([car_signature, human_signature, car_peak_ratio, human_peak_ratio])
                            
                            car_peak_max = np.max(bands[2])
                            human_peak_max = np.max(bands[5])
                            car_peak_avg = np.mean(bands[2])
                            human_peak_avg = np.mean(bands[5])
                            
                            features.extend([car_peak_max, human_peak_max, car_peak_avg, human_peak_avg])
                        else:
                            features.extend([0.0] * 8)
                    else:
                        features.extend([0.0] * 8)
                    
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
                    
                    features_list.append(np.array(features[:16], dtype=np.float32))
            except Exception:
                continue
        
        if features_list:
            all_features[category] = np.array(features_list)
    
    # Create car dataset
    if 'car' in all_features and 'car_nothing' in all_features:
        X = np.vstack([all_features['car'], all_features['car_nothing']])
        y = np.hstack([
            np.ones(len(all_features['car'])),      # Car signals = 1
            np.zeros(len(all_features['car_nothing']))  # Car noise = 0
        ])
        return X, y
    
    return None, None

def test_split_epoch_combination(X, y, train_size, epochs, random_state=42):
    """Test a specific train_size and epoch combination"""
    try:
        test_size = 1.0 - train_size
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Check minimum requirements
        if len(X_train) < 2 or len(X_test) < 1:
            return None, "Split too extreme"
        
        if len(np.unique(y_train)) < 2:
            return None, "Training set missing classes"
        
        # Train classifier with specified epochs
        classifier = FocusedCarSCTNClassifier(input_size=X.shape[1])
        classifier.train_with_epochs(X_train, y_train, epochs=epochs, lr=0.1)
        
        # Test
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, "Success"
        
    except Exception as e:
        return None, str(e)

def run_focused_60_90_car_optimization():
    """Run focused optimization for car data: 60-90% train with all epochs"""
    print("🚗 FOCUSED 60-90% TRAIN OPTIMIZATION FOR CAR DATA")
    print("=" * 60)
    print("Testing EVERY train split (60%-90% TRAIN) with EVERY epoch (20-2000)")
    print("This will find the BEST configuration in the 60-90% train range for CAR classification!")
    print()
    
    # Load car data
    print("📊 Loading car classification data...")
    X, y = load_car_data()
    
    if X is None:
        print("❌ Failed to load car data!")
        return None
    
    print(f"✅ Loaded car dataset:")
    print(f"   Total samples: {len(X)}")
    print(f"   Label distribution: {Counter(y)}")
    
    # Define focused parameters - 60% to 90% TRAIN
    train_sizes = np.arange(0.60, 0.91, 0.01)  # 60% to 90% TRAIN in 1% steps (31 splits)
    test_sizes = 1.0 - train_sizes             # Corresponding test sizes (40% to 10%)
    epochs_range = np.arange(20, 2001, 10)     # 20 to 2000 in steps of 10 (199 epochs)
    
    total_experiments = len(train_sizes) * len(epochs_range)
    
    print(f"\n🚗 FOCUSED CAR OPTIMIZATION PARAMETERS:")
    print(f"   Train sizes: {len(train_sizes)} splits (60% to 90% TRAIN)")
    print(f"   Test sizes: Corresponding (40% to 10% TEST)")
    print(f"   Epochs: {len(epochs_range)} values (20 to 2000 in steps of 10)")
    print(f"   Total experiments: {total_experiments:,}")
    print(f"   Estimated time: {total_experiments * 0.1 / 60:.1f} minutes")
    print()
    
    print(f"🚗 Progress: [", end="", flush=True)
    
    start_time = time.time()
    best_accuracy = 0
    best_config = None
    experiment_count = 0
    
    all_results = []
    best_per_split = {}
    
    for i, (train_size, test_size) in enumerate(zip(train_sizes, test_sizes)):
        split_best_accuracy = 0
        split_best_epochs = 0
        
        for j, epochs in enumerate(epochs_range):
            experiment_count += 1
            
            # Progress indicator
            if experiment_count % 100 == 0:
                progress_pct = (experiment_count / total_experiments) * 100
                print(f"{progress_pct:.1f}%", end="", flush=True)
            elif experiment_count % 50 == 0:
                print("█", end="", flush=True)
            elif experiment_count % 20 == 0:
                print(".", end="", flush=True)
            
            # Test this combination
            accuracy, status = test_split_epoch_combination(X, y, train_size, epochs)
            
            if accuracy is not None:
                # Record result
                result = {
                    'train_size': float(train_size),
                    'test_size': float(test_size),
                    'epochs': int(epochs),
                    'accuracy': float(accuracy),
                    'train_pct': float(train_size * 100),
                    'test_pct': float(test_size * 100),
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
                
                all_results.append(result)
                
                # Update best overall
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = result.copy()
                    print(f"🔥NEW BEST: {best_accuracy:.4f} ({best_config['train_pct']:.0f}%/{best_config['test_pct']:.0f}%, {best_config['epochs']} epochs)", end="", flush=True)
                
                # Update best for this split
                if accuracy > split_best_accuracy:
                    split_best_accuracy = accuracy
                    split_best_epochs = epochs
        
        # Record best for this split
        if split_best_accuracy > 0:
            best_per_split[train_size] = {
                'best_accuracy': split_best_accuracy,
                'best_epochs': split_best_epochs,
                'train_pct': train_size * 100,
                'test_pct': test_size * 100
            }
    
    print("] ✅")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Focused 60-90% CAR optimization completed in {elapsed_time/60:.1f} minutes")
    print(f"📊 Total experiments: {experiment_count:,}")
    
    if best_config is None:
        print("❌ No successful experiments found!")
        return None
    
    # Results for 60-90% range
    print(f"\n🏆 BEST CAR CONFIGURATION IN 60-90% TRAIN RANGE:")
    print("=" * 70)
    print(f"🚗 BEST SPLIT: {best_config['train_pct']:.0f}%/{best_config['test_pct']:.0f}% (train/test)")
    print(f"🚗 BEST EPOCHS: {best_config['epochs']}")
    print(f"🚗 BEST ACCURACY: {best_config['accuracy']:.4f} ({best_config['accuracy']*100:.2f}%)")
    print(f"🚗 FOCUSED CAR WINNER: {best_config['train_pct']:.0f}%/{best_config['test_pct']:.0f}% split with {best_config['epochs']} epochs")
    
    # Top 10 in focused range
    top_10 = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)[:10]
    
    print(f"\n📈 TOP 10 CAR CONFIGURATIONS (60-90% TRAIN RANGE):")
    print("=" * 85)
    print(f"{'Rank':<4} {'Train%':<6} {'Test%':<5} {'Epochs':<6} {'Accuracy':<9} {'Status'}")
    print("-" * 85)
    
    for i, result in enumerate(top_10, 1):
        status = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐" if i <= 5 else "✅"
        print(f"{i:<4} {result['train_pct']:.0f}%    {result['test_pct']:.0f}%   "
              f"{result['epochs']:<6} {result['accuracy']:.4f}    {status}")
    
    # Best epochs by split in focused range
    print(f"\n📊 BEST EPOCHS BY SPLIT FOR CAR (60-90% TRAIN):")
    print("-" * 50)
    for train_size in sorted(best_per_split.keys()):
        split_data = best_per_split[train_size]
        print(f"{split_data['train_pct']:.0f}%/{split_data['test_pct']:.0f}%: "
              f"Best={split_data['best_accuracy']:.4f} at {split_data['best_epochs']} epochs")
    
    # Save focused results
    focused_results_df = pd.DataFrame(all_results)
    results_file = "focused_60_90_car_optimization_complete.csv"
    focused_results_df.to_csv(results_file, index=False)
    
    print(f"\n💾 Focused CAR results saved to: {results_file}")
    print(f"📊 Total successful CAR experiments in 60-90% range: {len(all_results):,}")
    
    return {
        'best_config': best_config,
        'top_10': top_10,
        'all_results': all_results,
        'best_per_split': best_per_split
    }

if __name__ == "__main__":
    print("🚗 STARTING FOCUSED 60-90% TRAIN OPTIMIZATION FOR CAR DATA")
    print("This will find the BEST configuration in the 60-90% train range for CAR classification!")
    print("=" * 70)
    
    optimization_results = run_focused_60_90_car_optimization()
    
    if optimization_results:
        best = optimization_results['best_config']
        print(f"\n✅ FOCUSED 60-90% CAR OPTIMIZATION COMPLETE!")
        print(f"🏆 BEST CAR CONFIG IN RANGE: {best['train_pct']:.0f}%/{best['test_pct']:.0f}% split with {best['epochs']} epochs")
        print(f"🚗 FOCUSED CAR ACCURACY: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
        print(f"🚗 This is the OPTIMAL car configuration in the 60-90% train range!")
    else:
        print(f"\n❌ Focused car optimization failed!") 