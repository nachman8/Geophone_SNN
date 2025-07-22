#!/usr/bin/env python3
"""
Improved SCTN Geophone Classifier
Focusing on key spikegram patterns with proper sctnN library usage
"""

import numpy as np
import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import sctnN components
from sctnN.spiking_network import SpikingNetwork
from sctnN.layers import SCTNLayer
from sctnN.spiking_neuron import create_SCTN, BINARY, IDENTITY

class AdvancedPatternExtractor:
    """
    Advanced pattern extraction based on actual spikegram analysis
    Focuses on the key distinguishing features observed in the data
    """
    
    def __init__(self):
        # Band indices for 8-band system
        self.band_names = [
            'LOW_FREQ',      # 0: 20-30 Hz
            'CAR_APPROACH',  # 1: 30-34 Hz  
            'CAR_PEAK',      # 2: 34-40 Hz
            'CAR_TAIL',      # 3: 40-48 Hz
            'MID_GAP',       # 4: 48-60 Hz
            'HUMAN_PEAK',    # 5: 60-70 Hz
            'HUMAN_TAIL',    # 6: 70-85 Hz
            'HIGH_FREQ'      # 7: 85-100 Hz
        ]
        
        # Key signal patterns observed in spikegrams
        self.car_signature_bands = [1, 2, 3]      # 30-48 Hz (red vertical lines)
        self.human_signature_bands = [5, 6]       # 60-85 Hz (burst patterns)  
        self.noise_bands = [0, 7]                 # Low and high freq noise
        
        print(f"🔬 AdvancedPatternExtractor initialized")
        print(f"   🚗 Car signature: {self.car_signature_bands} (30-48 Hz)")
        print(f"   👤 Human signature: {self.human_signature_bands} (60-85 Hz)")
    
    def extract_key_features(self, spikes_bands_spectrogram, duration, signal_type):
        """
        Extract the most discriminative features based on spikegram patterns
        """
        features = []
        
        # Ensure we have the expected 8 bands
        if spikes_bands_spectrogram.shape[0] != 8:
            print(f"⚠️ Expected 8 bands, got {spikes_bands_spectrogram.shape[0]}")
            return np.zeros(20)  # Return default feature vector
        
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        # 1. SIGNAL SIGNATURE STRENGTH
        car_signature = np.sum(spikes_bands_spectrogram[self.car_signature_bands, :])
        human_signature = np.sum(spikes_bands_spectrogram[self.human_signature_bands, :])
        total_energy = np.sum(spikes_bands_spectrogram)
        
        # Signature ratios (most important features)
        car_ratio = car_signature / (total_energy + 1e-10)
        human_ratio = human_signature / (total_energy + 1e-10)
        
        features.extend([car_ratio, human_ratio])
        
        # 2. TEMPORAL PATTERN ANALYSIS
        # Car pattern: periodic vertical lines every ~50 seconds
        car_temporal = np.sum(spikes_bands_spectrogram[self.car_signature_bands, :], axis=0)
        car_periodicity = self._detect_periodicity(car_temporal, expected_period=50)
        car_consistency = np.std(car_temporal) / (np.mean(car_temporal) + 1e-10)
        
        # Human pattern: sporadic bursts
        human_temporal = np.sum(spikes_bands_spectrogram[self.human_signature_bands, :], axis=0)
        human_burstiness = self._detect_burstiness(human_temporal)
        human_sparsity = np.sum(human_temporal > 0) / len(human_temporal)
        
        features.extend([car_periodicity, car_consistency, human_burstiness, human_sparsity])
        
        # 3. FREQUENCY BAND DOMINANCE
        for band_idx in range(n_bands):
            band_energy = np.sum(spikes_bands_spectrogram[band_idx, :])
            band_dominance = band_energy / (total_energy + 1e-10)
            features.append(band_dominance)
        
        # 4. SIGNAL QUALITY METRICS
        noise_level = np.sum(spikes_bands_spectrogram[self.noise_bands, :])
        signal_level = car_signature + human_signature
        snr = signal_level / (noise_level + 1e-10)
        
        # Activity concentration
        overall_activity = np.sum(spikes_bands_spectrogram, axis=0)
        activity_peaks = np.sum(overall_activity > np.mean(overall_activity) + 2*np.std(overall_activity))
        activity_concentration = activity_peaks / len(overall_activity)
        
        features.extend([snr, activity_concentration])
        
        # 5. PATTERN-SPECIFIC FEATURES
        if signal_type == 'car':
            # Car-specific: look for consistent mid-frequency activity
            car_bands_data = spikes_bands_spectrogram[self.car_signature_bands, :]
            car_uniformity = 1.0 / (1.0 + np.std(np.mean(car_bands_data, axis=1)))
            car_persistence = np.sum(np.mean(car_bands_data, axis=0) > 0) / n_time_bins
            features.extend([car_uniformity, car_persistence])
        else:
            # Human-specific: look for burst characteristics  
            human_bands_data = spikes_bands_spectrogram[self.human_signature_bands, :]
            human_peak_intensity = np.max(human_bands_data)
            human_event_count = self._count_events(np.sum(human_bands_data, axis=0))
            features.extend([human_peak_intensity, human_event_count])
        
        return np.array(features)
    
    def _detect_periodicity(self, signal, expected_period=50):
        """Detect periodicity in signal (for car detection)"""
        if len(signal) < expected_period * 2:
            return 0
        
        # Check autocorrelation at expected period
        autocorr = np.correlate(signal, signal, mode='full')
        center = len(autocorr) // 2
        
        if center + expected_period < len(autocorr):
            periodic_strength = autocorr[center + expected_period] / autocorr[center]
        else:
            periodic_strength = 0
        
        return max(0, periodic_strength)
    
    def _detect_burstiness(self, signal):
        """Detect burst patterns (for human detection)"""
        if len(signal) == 0:
            return 0
        
        # Define burst threshold
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        
        # Count transitions from below to above threshold
        above_threshold = signal > threshold
        bursts = np.diff(np.concatenate([[False], above_threshold]))
        burst_count = np.sum(bursts == True)
        
        # Normalize by signal length
        burstiness = burst_count / len(signal) * 100
        
        return burstiness
    
    def _count_events(self, signal):
        """Count discrete events in signal"""
        if len(signal) == 0:
            return 0
        
        threshold = np.mean(signal) + np.std(signal)
        events = signal > threshold
        
        # Count separate event regions
        event_starts = np.diff(np.concatenate([[False], events]))
        event_count = np.sum(event_starts == True)
        
        return event_count


class SimpleSctnClassifier:
    """
    Simplified SCTN classifier focusing on stable learning
    Based on getting_started.ipynb patterns
    """
    
    def __init__(self, n_features):
        self.n_features = n_features
        self.network = None
        self.is_trained = False
        
        # Conservative STDP parameters for stable learning
        self.clk_freq = 153600
        self.A_LTP = 0.0001     # Reduced learning rate
        self.A_LTD = -0.00005   # Reduced learning rate
        self.tau = 1000         # Simplified time constant
        
        print(f"🧠 SimpleSctnClassifier: {n_features} features")
        print(f"   🔧 Conservative STDP: A_LTP={self.A_LTP}, A_LTD={self.A_LTD}")
    
    def _build_simple_network(self):
        """Build simplified SCTN network"""
        self.network = SpikingNetwork()
        
        # Single hidden layer with fewer neurons
        hidden_size = 6  # Small, stable network
        
        # Input layer  
        input_neurons = []
        for i in range(self.n_features):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY  
            input_neurons.append(neuron)
        self.network.add_layer(SCTNLayer(input_neurons))
        
        # Hidden layer
        hidden_neurons = []
        for i in range(hidden_size):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.5, 2.0, self.n_features).astype(np.float64)
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 20  # Lower threshold
            neuron.leakage_factor = 1
            neuron.leakage_period = 1
            neuron.theta = -0.05
            
            # Conservative STDP
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=10,  # Lower weight bounds
                wmin=0
            )
            
            hidden_neurons.append(neuron)
        self.network.add_layer(SCTNLayer(hidden_neurons))
        
        # Output layer (2 neurons)
        output_neurons = []
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.5, 1.5, hidden_size).astype(np.float64)
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 10
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.theta = -2
            neuron.label = f"class_{i}"
            
            hidden_neurons.append(neuron)
        self.network.add_layer(SCTNLayer(output_neurons))
        
        print(f"🏗️ Built simple network: {self.n_features} → {hidden_size} → 2")
    
    def train_with_supervised_stdp(self, X, y, n_epochs=20):
        """Train with supervised STDP approach"""
        if self.network is None:
            self._build_simple_network()
        
        # Normalize features to [0, 1]
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
        
        training_accuracies = []
        
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            
            for idx in indices:
                features = X_norm[idx]
                target = int(y[idx])
                
                # Simple spike encoding: feature values as spike probabilities
                spike_pattern = (np.random.random(self.n_features) < features).astype(int)
                
                # Process through network
                self.network.reset_input()
                
                outputs = []
                for _ in range(50):  # Process for multiple time steps
                    output = self.network.input(spike_pattern)
                    outputs.append(output)
                
                # Get prediction
                total_output = np.sum(outputs, axis=0)
                prediction = np.argmax(total_output) if np.sum(total_output) > 0 else 0
                
                if prediction == target:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            training_accuracies.append(accuracy)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: {accuracy:.1%}")
        
        self.is_trained = True
        return training_accuracies
    
    def predict(self, X):
        """Predict using trained network"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Normalize features
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
        
        predictions = []
        
        for features in X_norm:
            # Encode features
            spike_pattern = (np.random.random(self.n_features) < features).astype(int)
            
            # Process through network
            self.network.reset_input()
            
            outputs = []
            for _ in range(50):
                output = self.network.input(spike_pattern)
                outputs.append(output)
            
            # Get prediction
            total_output = np.sum(outputs, axis=0)
            prediction = np.argmax(total_output) if np.sum(total_output) > 0 else 0
            predictions.append(prediction)
        
        return np.array(predictions)


def load_saved_chunks(chunks_base_dir):
    """Load existing chunk files"""
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
            
            chunk_data[signal_type] = {
                'index': chunk_index,
                'chunks': chunks
            }
            
            print(f"   ✅ {signal_type}: {len(chunks)} chunks")
    
    return chunk_data


def main():
    """Main execution"""
    print("🚀 IMPROVED SCTN GEOPHONE CLASSIFIER")
    print("=" * 50)
    print("Focusing on key spikegram patterns with stable learning")
    print()
    
    # Initialize extractor
    extractor = AdvancedPatternExtractor()
    
    # Load data
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    chunk_data = load_saved_chunks(chunks_dir)
    
    if not chunk_data:
        print("❌ No data found")
        return
    
    # Extract features focusing on key patterns
    print("\n🔬 EXTRACTING KEY DISCRIMINATIVE FEATURES")
    print("-" * 40)
    
    all_features = []
    all_labels = []
    
    for signal_type, data in chunk_data.items():
        is_detection = not signal_type.endswith('_nothing')
        label = 1 if is_detection else 0
        
        base_type = signal_type.replace('_nothing', '')
        
        for chunk_idx, chunk in enumerate(data['chunks']):
            if 'spikes_bands_spectrogram' not in chunk:
                continue
            
            spikes_bands_spectrogram = chunk['spikes_bands_spectrogram']
            duration = chunk.get('duration', 120)
            
            # Extract key features
            features = extractor.extract_key_features(
                spikes_bands_spectrogram, 
                duration, 
                base_type
            )
            
            all_features.append(features)
            all_labels.append(label)
        
        print(f"   {signal_type}: {len(data['chunks'])} chunks processed")
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n📊 DATASET SUMMARY")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Detections: {np.sum(y == 1)}")
    print(f"   Nothing: {np.sum(y == 0)}")
    
    # Check feature quality
    print(f"\n🔍 FEATURE ANALYSIS")
    for i in range(min(5, X.shape[1])):
        detection_vals = X[y == 1, i]
        nothing_vals = X[y == 0, i]
        
        if len(detection_vals) > 0 and len(nothing_vals) > 0:
            separation = abs(np.mean(detection_vals) - np.mean(nothing_vals))
            print(f"   Feature {i}: separation = {separation:.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Compare multiple approaches
    print(f"\n🧠 TRAINING CLASSIFIERS")
    print("-" * 40)
    
    results = {}
    
    # 1. Random Forest baseline
    print("🌲 Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['RandomForest'] = rf_acc
    print(f"   Accuracy: {rf_acc:.1%}")
    
    # 2. Simple SCTN Classifier
    print("🧠 Simple SCTN classifier...")
    sctn_clf = SimpleSctnClassifier(n_features=X.shape[1])
    sctn_history = sctn_clf.train_with_supervised_stdp(X_train, y_train, n_epochs=25)
    sctn_pred = sctn_clf.predict(X_test)
    sctn_acc = accuracy_score(y_test, sctn_pred)
    results['SCTN'] = sctn_acc
    
    print(f"\n📊 FINAL RESULTS")
    print("-" * 40)
    
    for method, accuracy in results.items():
        print(f"{method}: {accuracy:.1%}")
    
    # Detailed analysis for best method
    best_method = max(results, key=results.get)
    best_acc = results[best_method]
    
    print(f"\n🎯 BEST METHOD: {best_method} ({best_acc:.1%})")
    
    if best_method == 'SCTN':
        pred = sctn_pred
        print(f"\nSCTN Training: {sctn_history[0]:.1%} → {sctn_history[-1]:.1%}")
    else:
        pred = rf_pred
    
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=['Nothing', 'Detection']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    
    if best_acc >= 0.80:
        print("🎉 EXCELLENT! High accuracy achieved!")
    elif best_acc >= 0.70:
        print("✅ GOOD! Significant improvement!")
    else:
        print("⚠️ Needs more optimization")
    
    return results


if __name__ == "__main__":
    results = main()
