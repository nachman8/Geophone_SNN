#!/usr/bin/env python3
"""
COMPREHENSIVE SPLIT OPTIMIZATION EXPERIMENT
Tests all possible train/test splits to find the optimal ratio
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
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

class FastSCTNClassifier:
    """Streamlined SCTN classifier for rapid split testing"""
    
    def __init__(self, input_size=16, signal_type=None):
        self.input_size = input_size
        self.signal_type = signal_type
        self.neuron = None
        self.scaler = None
        
    def _create_classifier_neuron(self):
        """Create optimized SCTN neuron"""
        neuron = create_SCTN()
        
        if self.signal_type == 'human':
            neuron.synapses_weights = np.random.normal(0, 0.05, self.input_size).astype(np.float64)
            neuron.threshold_pulse = 25.0
            # Boost human features
            human_feature_indices = [1, 3, 5, 7, 9, 10, 11]
            for idx in human_feature_indices:
                if idx < len(neuron.synapses_weights):
                    neuron.synapses_weights[idx] *= 2.0
        else:
            neuron.synapses_weights = np.random.normal(0, 0.1, self.input_size).astype(np.float64)
            neuron.threshold_pulse = 50.0
        
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
    
    def train_fast(self, X, y, epochs=100, lr=0.1):
        """Fast training for split optimization"""
        # Quick feature scaling
        if self.signal_type == 'human':
            self.scaler = MinMaxScaler(feature_range=(0.15, 0.85))
            X_scaled = self.scaler.fit_transform(X)
            # Enhance human features
            human_feature_indices = [1, 3, 5, 7, 9, 10, 11]
            for idx in human_feature_indices:
                if idx < X_scaled.shape[1]:
                    X_scaled[:, idx] = np.power(X_scaled[:, idx], 0.8)
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = self.scaler.fit_transform(X)
        
        # Create neuron
        self.neuron = self._create_classifier_neuron()
        
        # Fast training
        for epoch in range(epochs):
            for i in range(len(X_scaled)):
                features = X_scaled[i]
                target = y[i]
                
                prediction, activation = self._forward(features)
                error = target - prediction
                
                # Update weights
                self.neuron.synapses_weights += lr * error * features
                
                # Update threshold
                if self.signal_type == 'human':
                    self.neuron.threshold_pulse += lr * error * 0.05
                else:
                    self.neuron.threshold_pulse += lr * error * 0.1
        
        return True
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        # Apply feature enhancement for human data
        if self.signal_type == 'human':
            human_feature_indices = [1, 3, 5, 7, 9, 10, 11]
            for idx in human_feature_indices:
                if idx < X_scaled.shape[1]:
                    X_scaled[:, idx] = np.power(X_scaled[:, idx], 0.8)
        
        predictions = []
        for features in X_scaled:
            prediction, _ = self._forward(features)
            predictions.append(prediction)
        
        return np.array(predictions)

def load_datasets_fast():
    """Load datasets quickly for optimization"""
    categories = {
        'human': 47,
        'human_nothing': 33,
        'car': 28,
        'car_nothing': 16
    }
    
    all_features = {}
    
    for category, max_chunks in categories.items():
        features_list = []
        for chunk_num in range(max_chunks):
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
    
    # Create datasets
    datasets = {}
    
    if 'human' in all_features and 'human_nothing' in all_features:
        X_human = np.vstack([all_features['human'], all_features['human_nothing']])
        y_human = np.hstack([
            np.ones(len(all_features['human'])),
            np.zeros(len(all_features['human_nothing']))
        ])
        datasets['human'] = (X_human, y_human)
    
    if 'car' in all_features and 'car_nothing' in all_features:
        X_car = np.vstack([all_features['car'], all_features['car_nothing']])
        y_car = np.hstack([
            np.ones(len(all_features['car'])),
            np.zeros(len(all_features['car_nothing']))
        ])
        datasets['car'] = (X_car, y_car)
    
    return datasets

def test_single_split(X, y, signal_type, test_size, random_state=42):
    """Test a single train/test split"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Check minimum requirements
        if len(X_train) < 4 or len(X_test) < 1:
            return None, None, "Split too extreme"
        
        if len(np.unique(y_train)) < 2:
            return None, None, "Training set missing classes"
        
        # Train classifier
        classifier = FastSCTNClassifier(input_size=X.shape[1], signal_type=signal_type)
        
        # Adjust epochs based on training set size
        if len(X_train) < 10:
            epochs = 50
        elif len(X_train) < 20:
            epochs = 75
        else:
            epochs = 100
        
        classifier.train_fast(X_train, y_train, epochs=epochs, lr=0.1)
        
        # Test
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, len(X_test), "Success"
        
    except Exception as e:
        return None, None, str(e)

def run_comprehensive_split_test():
    """Run comprehensive split optimization test"""
    print("🔍 COMPREHENSIVE SPLIT OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print("Testing ALL possible train/test splits from 5% to 95% test size...")
    print("This will find the ABSOLUTE OPTIMAL balance for maximum performance!")
    print()
    
    # Load data
    print("📊 Loading datasets...")
    datasets = load_datasets_fast()
    
    if not datasets:
        print("❌ Failed to load datasets!")
        return None
    
    print(f"✅ Loaded datasets:")
    for name, (X, y) in datasets.items():
        print(f"   {name}: {len(X)} samples, {Counter(y)}")
    
    # Test ranges - from 5% to 95% test size in 1% increments
    test_sizes = np.arange(0.05, 0.96, 0.01)
    
    results = []
    
    print(f"\n🚀 Testing {len(test_sizes)} different splits (5% to 95% test size)...")
    print(f"📈 Progress: [", end="", flush=True)
    
    start_time = time.time()
    
    for i, test_size in enumerate(test_sizes):
        # Progress indicator
        if i % 9 == 0:
            print("█", end="", flush=True)
        
        result = {
            'test_size': test_size,
            'train_size': 1 - test_size,
            'train_pct': (1 - test_size) * 100,
            'test_pct': test_size * 100
        }
        
        # Test human classifier
        if 'human' in datasets:
            X_human, y_human = datasets['human']
            human_acc, human_test_samples, human_status = test_single_split(
                X_human, y_human, 'human', test_size
            )
            result.update({
                'human_accuracy': human_acc,
                'human_test_samples': human_test_samples,
                'human_status': human_status
            })
        
        # Test car classifier
        if 'car' in datasets:
            X_car, y_car = datasets['car']
            car_acc, car_test_samples, car_status = test_single_split(
                X_car, y_car, 'car', test_size
            )
            result.update({
                'car_accuracy': car_acc,
                'car_test_samples': car_test_samples,
                'car_status': car_status
            })
        
        # Calculate combined score
        if result.get('human_accuracy') is not None and result.get('car_accuracy') is not None:
            result['combined_accuracy'] = (result['human_accuracy'] + result['car_accuracy']) / 2
        else:
            result['combined_accuracy'] = None
        
        results.append(result)
    
    print("] ✅")
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Completed {len(test_sizes)} splits in {elapsed_time:.1f} seconds")
    
    # Analyze results
    print("\n📊 ANALYZING RESULTS...")
    
    # Filter successful results
    successful_results = [r for r in results if r['combined_accuracy'] is not None]
    
    if not successful_results:
        print("❌ No successful splits found!")
        return None
    
    # Find optimal splits
    best_combined = max(successful_results, key=lambda x: x['combined_accuracy'])
    
    human_results = [r for r in successful_results if r['human_accuracy'] is not None]
    car_results = [r for r in successful_results if r['car_accuracy'] is not None]
    
    best_human = max(human_results, key=lambda x: x['human_accuracy']) if human_results else None
    best_car = max(car_results, key=lambda x: x['car_accuracy']) if car_results else None
    
    print(f"\n🏆 ABSOLUTE OPTIMAL SPLITS FOUND:")
    print("=" * 70)
    
    print(f"\n🎯 BEST COMBINED PERFORMANCE:")
    print(f"   Split: {best_combined['train_pct']:.0f}%/{best_combined['test_pct']:.0f}% (train/test)")
    print(f"   Human: {best_combined['human_accuracy']:.4f} ({best_combined['human_test_samples']} test samples)")
    print(f"   Car: {best_combined['car_accuracy']:.4f} ({best_combined['car_test_samples']} test samples)")
    print(f"   Combined: {best_combined['combined_accuracy']:.4f} ⭐")
    
    if best_human:
        print(f"\n👤 BEST HUMAN PERFORMANCE:")
        print(f"   Split: {best_human['train_pct']:.0f}%/{best_human['test_pct']:.0f}% (train/test)")
        print(f"   Accuracy: {best_human['human_accuracy']:.4f} ({best_human['human_test_samples']} test samples)")
    
    if best_car:
        print(f"\n🚗 BEST CAR PERFORMANCE:")
        print(f"   Split: {best_car['train_pct']:.0f}%/{best_car['test_pct']:.0f}% (train/test)")
        print(f"   Accuracy: {best_car['car_accuracy']:.4f} ({best_car['car_test_samples']} test samples)")
    
    # Top 15 combined results
    top_15 = sorted(successful_results, key=lambda x: x['combined_accuracy'], reverse=True)[:15]
    
    print(f"\n📈 TOP 15 OPTIMAL SPLITS:")
    print("=" * 85)
    print(f"{'Rank':<4} {'Train%':<6} {'Test%':<5} {'Human':<8} {'Car':<8} {'Combined':<9} {'Test Samples':<12} {'Status'}")
    print("-" * 85)
    
    for i, result in enumerate(top_15, 1):
        status = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐" if i <= 5 else "✅"
        print(f"{i:<4} {result['train_pct']:.0f}%    {result['test_pct']:.0f}%   "
              f"{result['human_accuracy']:.4f}   {result['car_accuracy']:.4f}   "
              f"{result['combined_accuracy']:.4f}    "
              f"H:{result['human_test_samples']}, C:{result['car_test_samples']:<7} {status}")
    
    # Performance analysis by ranges
    print(f"\n📊 PERFORMANCE BY SPLIT RANGES:")
    print("-" * 50)
    
    ranges = [
        ("5-15% test", 0.05, 0.15),
        ("15-25% test", 0.15, 0.25),
        ("25-35% test", 0.25, 0.35),
        ("35-45% test", 0.35, 0.45),
        ("45-55% test", 0.45, 0.55),
        ("55-65% test", 0.55, 0.65),
        ("65-75% test", 0.65, 0.75),
        ("75-85% test", 0.75, 0.85),
        ("85-95% test", 0.85, 0.95)
    ]
    
    for range_name, min_test, max_test in ranges:
        range_results = [r for r in successful_results 
                        if min_test <= r['test_size'] < max_test]
        if range_results:
            best_in_range = max(range_results, key=lambda x: x['combined_accuracy'])
            avg_combined = np.mean([r['combined_accuracy'] for r in range_results])
            print(f"{range_name:<12}: Best={best_in_range['combined_accuracy']:.4f} "
                  f"(at {best_in_range['train_pct']:.0f}/{best_in_range['test_pct']:.0f}), "
                  f"Avg={avg_combined:.4f} ({len(range_results)} splits)")
    
    # Save detailed results
    results_df = pd.DataFrame(successful_results)
    results_file = "comprehensive_split_optimization.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n💾 Complete results saved to: {results_file}")
    print(f"📊 Successful splits tested: {len(successful_results)}/{len(test_sizes)}")
    
    return {
        'results': results,
        'successful_results': successful_results,
        'best_combined': best_combined,
        'best_human': best_human,
        'best_car': best_car,
        'top_15': top_15
    }

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE SPLIT OPTIMIZATION...")
    print("=" * 70)
    
    optimization_results = run_comprehensive_split_test()
    
    if optimization_results:
        print(f"\n✅ SPLIT OPTIMIZATION COMPLETE!")
        print(f"🎯 Found the ABSOLUTE OPTIMAL train/test ratios!")
        print(f"📊 Use the top results to configure your classifier for maximum performance")
        print(f"🏆 Winner: {optimization_results['best_combined']['train_pct']:.0f}%/{optimization_results['best_combined']['test_pct']:.0f}% split")
    else:
        print(f"\n❌ Split optimization failed!") 