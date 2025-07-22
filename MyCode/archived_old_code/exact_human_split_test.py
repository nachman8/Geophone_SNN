#!/usr/bin/env python3
"""
EXACT HUMAN SPLIT TEST: 50% to 98% test size in 1% increments
Exactly what the user requested
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

# Add the directory CONTAINING sctnN to your Python path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY

CHUNKED_OUTPUT_DIR = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"

class HumanSCTNClassifier:
    """Optimized SCTN classifier specifically for human signals"""
    
    def __init__(self, input_size=16):
        self.input_size = input_size
        self.neuron = None
        self.scaler = None
        
    def _create_human_neuron(self):
        """Create optimized SCTN neuron for human signals"""
        neuron = create_SCTN()
        
        # Human-optimized parameters
        neuron.synapses_weights = np.random.normal(0, 0.05, self.input_size).astype(np.float64)
        neuron.threshold_pulse = 25.0
        
        # Boost human-specific features
        human_feature_indices = [1, 3, 5, 7, 9, 10, 11]
        car_suppression_indices = [0, 2, 4, 6]
        
        for idx in human_feature_indices:
            if idx < len(neuron.synapses_weights):
                neuron.synapses_weights[idx] *= 2.0
        
        for idx in car_suppression_indices:
            if idx < len(neuron.synapses_weights):
                neuron.synapses_weights[idx] *= 0.7
        
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
    
    def train_human_optimized(self, X, y, epochs=150, lr=0.12):
        """Human-optimized training"""
        # Human-specific feature scaling
        self.scaler = MinMaxScaler(feature_range=(0.15, 0.85))
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature enhancement for human data
        human_feature_indices = [1, 3, 5, 7, 9, 10, 11]
        for idx in human_feature_indices:
            if idx < X_scaled.shape[1]:
                X_scaled[:, idx] = X_scaled[:, idx] * 1.2
        
        # Create human-optimized neuron
        self.neuron = self._create_human_neuron()
        
        # Adaptive learning rate for human patterns
        for epoch in range(epochs):
            # Phase-based learning rate
            if epoch <= 30:
                current_lr = lr * 1.5
            elif epoch <= 80:
                current_lr = lr
            else:
                current_lr = lr * (0.90 ** ((epoch - 80) // 10))
            
            for i in range(len(X_scaled)):
                features = X_scaled[i]
                target = y[i]
                
                prediction, activation = self._forward(features)
                error = target - prediction
                
                # Update weights
                self.neuron.synapses_weights += current_lr * error * features
                
                # Conservative threshold update for human data
                self.neuron.threshold_pulse += current_lr * error * 0.05
        
        return True
    
    def predict(self, X):
        """Make predictions with human-specific preprocessing"""
        X_scaled = self.scaler.transform(X)
        
        # Apply same feature enhancement as during training
        human_feature_indices = [1, 3, 5, 7, 9, 10, 11]
        for idx in human_feature_indices:
            if idx < X_scaled.shape[1]:
                X_scaled[:, idx] = X_scaled[:, idx] * 1.2
        
        predictions = []
        for features in X_scaled:
            prediction, _ = self._forward(features)
            predictions.append(prediction)
        
        return np.array(predictions)

def load_human_data():
    """Load only human classification data"""
    categories = ['human', 'human_nothing']
    chunk_counts = {'human': 47, 'human_nothing': 33}
    
    all_features = {}
    
    for category in categories:
        features_list = []
        for chunk_num in range(chunk_counts[category]):
            chunk_path = os.path.join(CHUNKED_OUTPUT_DIR, category, f"chunk_{chunk_num}", f"chunk_{chunk_num}_data.pkl")
            try:
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Extract 16 features quickly
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
    
    # Create human dataset
    if 'human' in all_features and 'human_nothing' in all_features:
        X = np.vstack([all_features['human'], all_features['human_nothing']])
        y = np.hstack([
            np.ones(len(all_features['human'])),
            np.zeros(len(all_features['human_nothing']))
        ])
        return X, y
    
    return None, None

def test_human_split(X, y, test_size, random_state=42):
    """Test a single train/test split for human classification"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Check minimum requirements
        if len(X_train) < 2 or len(X_test) < 1:
            return None, None, None, "Split too extreme"
        
        if len(np.unique(y_train)) < 2:
            return None, None, None, "Training set missing classes"
        
        # Train human classifier
        classifier = HumanSCTNClassifier(input_size=X.shape[1])
        
        # Adjust epochs based on training set size
        if len(X_train) < 10:
            epochs = 100
        elif len(X_train) < 20:
            epochs = 150
        else:
            epochs = 200
        
        classifier.train_human_optimized(X_train, y_train, epochs=epochs, lr=0.12)
        
        # Test
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, len(X_test), len(X_train), "Success"
        
    except Exception as e:
        return None, None, None, str(e)

def run_exact_test():
    """Run EXACTLY what user requested: 50% to 98% test size in 1% increments"""
    print("🎯 EXACT TEST: 50% to 98% test size in 1% increments")
    print("=" * 60)
    print("Testing EXACTLY as requested: every 1% from 50% to 98% test size")
    print()
    
    # Load human data
    print("📊 Loading human classification data...")
    X, y = load_human_data()
    
    if X is None:
        print("❌ Failed to load human data!")
        return None
    
    print(f"✅ Loaded human dataset:")
    print(f"   Total samples: {len(X)}")
    print(f"   Label distribution: {Counter(y)}")
    
    # EXACTLY what user requested: 50% to 98% in 1% increments
    test_sizes = np.arange(0.50, 0.99, 0.01)  # 50% to 98% in 1% steps
    
    print(f"\n🚀 Testing EXACTLY {len(test_sizes)} splits (50% to 98% test size)...")
    print(f"📈 Progress: [", end="", flush=True)
    
    results = []
    start_time = time.time()
    
    for i, test_size in enumerate(test_sizes):
        # Progress indicator
        if i % max(1, len(test_sizes)//30) == 0:
            print("█", end="", flush=True)
        
        accuracy, test_samples, train_samples, status = test_human_split(X, y, test_size)
        
        if accuracy is not None:
            results.append({
                'test_size': test_size,
                'train_size': 1 - test_size,
                'train_pct': (1 - test_size) * 100,
                'test_pct': test_size * 100,
                'accuracy': accuracy,
                'test_samples': test_samples,
                'train_samples': train_samples,
                'status': status
            })
    
    print("] ✅")
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Completed {len(test_sizes)} splits in {elapsed_time:.1f} seconds")
    
    if not results:
        print("❌ No successful splits found!")
        return None
    
    # Find the best results
    best_result = max(results, key=lambda x: x['accuracy'])
    
    # Get top 15 results
    top_15 = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:15]
    
    print(f"\n🏆 BEST HUMAN PERFORMANCE (50%-98% test range):")
    print("=" * 65)
    print(f"🎯 OPTIMAL SPLIT: {best_result['train_pct']:.1f}%/{best_result['test_pct']:.1f}% (train/test)")
    print(f"🎯 BEST ACCURACY: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"📊 Test samples: {best_result['test_samples']}")
    print(f"📊 Train samples: {best_result['train_samples']}")
    
    print(f"\n📈 TOP 15 RESULTS (50%-98% test range):")
    print("=" * 75)
    print(f"{'Rank':<4} {'Train%':<6} {'Test%':<5} {'Accuracy':<9} {'Test Samples':<12} {'Status'}")
    print("-" * 75)
    
    for i, result in enumerate(top_15, 1):
        status = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐" if i <= 5 else "✅"
        print(f"{i:<4} {result['train_pct']:.1f}%   {result['test_pct']:.1f}%  "
              f"{result['accuracy']:.4f}    {result['test_samples']:<12} {status}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = "exact_human_50_to_98_test.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n💾 Exact test results saved to: {results_file}")
    print(f"📊 Successful splits tested: {len(results)}/{len(test_sizes)}")
    
    # Show pattern analysis
    print(f"\n📊 PERFORMANCE PATTERN ANALYSIS:")
    print("-" * 50)
    accuracies = [r['accuracy'] for r in results]
    test_samples = [r['test_samples'] for r in results]
    
    print(f"Accuracy range: {min(accuracies):.4f} to {max(accuracies):.4f}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Test samples range: {min(test_samples)} to {max(test_samples)}")
    
    return {
        'results': results,
        'best_result': best_result,
        'top_15': top_15
    }

if __name__ == "__main__":
    print("🚀 EXACT HUMAN SPLIT TEST (as requested)")
    print("=" * 50)
    
    optimization_results = run_exact_test()
    
    if optimization_results:
        best = optimization_results['best_result']
        print(f"\n✅ EXACT TEST COMPLETE!")
        print(f"🏆 BEST in 50%-98% range: {best['train_pct']:.1f}%/{best['test_pct']:.1f}% split")
        print(f"🎯 BEST HUMAN ACCURACY: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    else:
        print(f"\n❌ Exact test failed!") 




    # עדכון משקולות
    self.neuron.synapses_weights += current_lr * error * features

    # עדכון סף (שונה לכל סוג אות)
    if self.signal_type == 'human':
        self.neuron.threshold_pulse += current_lr * error * 0.05
    else:
        self.neuron.threshold_pulse += current_lr * error * 0.1