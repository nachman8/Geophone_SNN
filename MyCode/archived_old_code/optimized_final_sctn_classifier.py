#!/usr/bin/env python3
"""
FINAL OPTIMIZED sctnN Classifier
Addresses training instability while using correct notebook parameters
"""

import numpy as np
import pickle
import os
import sys

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class OptimizedFinalSCTNClassifier:
    """
    FINAL OPTIMIZED sctnN classifier - combines all insights
    """
    
    def __init__(self, spike_duration=80):
        self.clk_freq = 1536000
        self.spike_duration = spike_duration
        self.network = None
        self.trained = False
        
        # OPTIMIZED parameters combining notebook + stability
        self.time_to_learn = 2.5e-3
        self.A_LTP = 0.0002     # Slightly lower for stability
        self.A_LTD = -0.00012   # Balanced ratio
        self.tau = self.clk_freq * self.time_to_learn / 2
        
        print(f"🚀 FINAL OPTIMIZED sctnN Classifier")
        print(f"   📈 Optimized learning: A_LTP={self.A_LTP}")
        print(f"   🎯 Stability + Performance focus")
    
    def create_network(self, n_features):
        print(f"🏗️  Creating OPTIMIZED FINAL network...")
        
        self.network = SpikingNetwork()
        
        # Input layer (simpler)
        input_neurons = []
        for i in range(n_features):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 5  # Lower threshold for sensitivity
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Hidden layer (optimized size)
        n_hidden = 12  # Smaller for stability with small dataset
        hidden_neurons = []
        
        for i in range(n_hidden):
            neuron = create_SCTN()
            # Better weight initialization
            neuron.synapses_weights = np.random.normal(0.5, 0.15, n_features).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.2, 0.8)
            
            # Optimized parameters for stability
            neuron.leakage_factor = 3  # Stronger leakage for stability
            neuron.leakage_period = 8
            neuron.theta = -8  # Lower threshold
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            
            # OPTIMIZED STDP
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=1.0,
                wmin=0.1  # Prevent weights from going to zero
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer (optimized supervised STDP)
        output_neurons = []
        
        for i in range(2):
            neuron = create_SCTN()
            # Conservative weight initialization
            neuron.synapses_weights = np.random.normal(0.4, 0.1, n_hidden).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.2, 0.6)
            
            # Stable parameters
            neuron.leakage_factor = 2
            neuron.leakage_period = 6
            neuron.theta = -6  # Lower threshold for easier firing
            neuron.activation_function = BINARY
            neuron.membrane_should_reset = True
            
            # OPTIMIZED supervised STDP
            neuron.set_supervised_stdp(
                A=0.02,  # Lower for stability
                tau=15.0,  # Longer for stability
                clk_freq=1000,
                wmax=2.0,  # Lower bounds
                wmin=0.1,
                desired_output=np.array([], dtype=np.int64)
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable logging
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ OPTIMIZED Network: {n_features} → {n_hidden} → 2")
        return self.network
    
    def convert_to_spikes(self, features):
        """Optimized spike conversion with deterministic component"""
        n_samples, n_features = features.shape
        spike_trains = np.zeros((n_samples, n_features, self.spike_duration), dtype=np.int8)
        
        for i, sample in enumerate(features):
            # Normalize to [0, 1]
            normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-10)
            
            for j, rate in enumerate(normalized):
                # Optimized spike rate (lower max for stability)
                spike_rate = rate * 25  # Max 25 Hz for stability
                
                # Mix deterministic and stochastic spikes for stability
                deterministic_spikes = int(spike_rate * 0.6)  # 60% deterministic
                stochastic_spikes = spike_rate - deterministic_spikes
                
                # Place deterministic spikes evenly
                if deterministic_spikes > 0:
                    spacing = self.spike_duration // deterministic_spikes
                    for k in range(deterministic_spikes):
                        spike_time = k * spacing + np.random.randint(0, spacing//2 + 1)
                        if spike_time < self.spike_duration:
                            spike_trains[i, j, spike_time] = 1
                
                # Add stochastic spikes
                spike_prob = stochastic_spikes / self.spike_duration
                random_spikes = np.random.random(self.spike_duration) < spike_prob
                spike_trains[i, j, :] = np.logical_or(spike_trains[i, j, :], random_spikes).astype(np.int8)
        
        return spike_trains
    
    def load_data(self, chunks_dir):
        """Load data with enhanced features"""
        X, y = [], []
        
        # Load car chunks
        car_dir = os.path.join(chunks_dir, "car")
        car_index_file = os.path.join(car_dir, "chunk_index.pkl")
        if os.path.exists(car_index_file):
            with open(car_index_file, 'rb') as f:
                car_index = pickle.load(f)
            
            for chunk_file in car_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    
                    # Enhanced features
                    features = []
                    for band_data in spikegram:
                        features.extend([
                            np.mean(band_data),
                            np.std(band_data),
                            np.max(band_data)
                        ])
                    
                    X.append(features)
                    y.append(0)  # Car = 0
        
        # Load nothing chunks
        nothing_dir = os.path.join(chunks_dir, "car_nothing")
        nothing_index_file = os.path.join(nothing_dir, "chunk_index.pkl")
        if os.path.exists(nothing_index_file):
            with open(nothing_index_file, 'rb') as f:
                nothing_index = pickle.load(f)
            
            for chunk_file in nothing_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    
                    features = []
                    for band_data in spikegram:
                        features.extend([
                            np.mean(band_data),
                            np.std(band_data),
                            np.max(band_data)
                        ])
                    
                    X.append(features)
                    y.append(1)  # Nothing = 1
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Enhanced data: {len(X)} samples, {X.shape[1]} features")
        if len(X) > 1:
            car_mean = np.mean(X[y == 0])
            nothing_mean = np.mean(X[y == 1])
            print(f"   Car mean: {car_mean:.2f}, Nothing mean: {nothing_mean:.2f}")
            print(f"   Discrimination ratio: {car_mean/nothing_mean:.2f}x")
        
        return X, y
    
    def train(self, X, y, n_epochs=50):
        """Optimized training with stability monitoring"""
        if self.network is None:
            self.create_network(X.shape[1])
        
        print("Converting to optimized spike trains...")
        X_spikes = self.convert_to_spikes(X)
        
        history = []
        stable_epochs = 0
        last_accuracy = 0
        
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_spikes))
            
            for idx in indices:
                spike_train = X_spikes[idx]
                target = y[idx]
                
                # Reset network
                self.network.reset_input()
                
                # Set supervised targets with optimized patterns
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target:
                            # Optimized spike timing patterns
                            if target == 0:  # Car
                                spike_times = list(range(15, self.spike_duration-15, 25))
                            else:  # Nothing
                                spike_times = list(range(20, self.spike_duration-20, 30))
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target should not spike
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spike train
                output_spikes = []
                for t in range(self.spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Prediction
                total_spikes = np.sum(output_spikes, axis=0)
                predicted = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 1
                
                if predicted == target:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            history.append(accuracy)
            
            # Stability monitoring
            if abs(accuracy - last_accuracy) < 0.05:
                stable_epochs += 1
            else:
                stable_epochs = 0
            
            last_accuracy = accuracy
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Accuracy = {accuracy:.1%}, Stable epochs: {stable_epochs}")
            
            # Early stopping if stable
            if stable_epochs >= 10 and epoch > 20:
                print(f"✅ Stable training achieved at epoch {epoch+1}")
                break
        
        self.trained = True
        
        # Enhanced learning analysis
        if len(history) > 5:
            initial_acc = np.mean(history[:3])  # Average of first 3 epochs
            final_acc = np.mean(history[-3:])   # Average of last 3 epochs
            improvement = final_acc - initial_acc
            max_acc = max(history)
            
            print(f"\n📈 COMPREHENSIVE LEARNING ANALYSIS:")
            print(f"   Initial accuracy: {initial_acc:.1%}")
            print(f"   Final accuracy: {final_acc:.1%}")
            print(f"   Peak accuracy: {max_acc:.1%}")
            print(f"   Net improvement: {improvement:+.1%}")
            
            if improvement > 0.1:
                print("   🔥 EXCELLENT LEARNING!")
            elif final_acc > 0.6:
                print("   ✅ GOOD PERFORMANCE!")
            else:
                print("   ⚠️  Limited performance")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Enhanced evaluation with confidence scoring"""
        X_test_spikes = self.convert_to_spikes(X_test)
        predictions = []
        confidences = []
        
        for spike_train in X_test_spikes:
            self.network.reset_input()
            
            output_spikes = []
            for t in range(self.spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            total_spikes = np.sum(output_spikes, axis=0)
            
            if np.sum(total_spikes) > 0:
                prediction = np.argmax(total_spikes)
                confidence = np.max(total_spikes) / np.sum(total_spikes)
            else:
                prediction = 1  # Default to nothing
                confidence = 0.5
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Car', 'Nothing'])
        cm = confusion_matrix(y_test, predictions)
        
        avg_confidence = np.mean(confidences)
        
        return accuracy, report, cm, avg_confidence

# Main execution
if __name__ == "__main__":
    print("🚀 FINAL OPTIMIZED sctnN CLASSIFIER")
    print("=" * 60)
    print("Combining correct parameters + stability optimizations")
    print()
    
    classifier = OptimizedFinalSCTNClassifier()
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    X, y = classifier.load_data(chunks_dir)
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"📊 Training on {len(X_train)} samples...")
        history = classifier.train(X_train, y_train, n_epochs=80)
        
        accuracy, report, cm, confidence = classifier.evaluate(X_test, y_test)
        
        print(f"\n🎯 FINAL OPTIMIZED RESULTS:")
        print(f"📊 Test Accuracy: {accuracy:.1%}")
        print(f"📊 Avg Confidence: {confidence:.1%}")
        print(f"\n📈 Classification Report:")
        print(report)
        print(f"\n🔥 Confusion Matrix:")
        print(cm)
        
        if accuracy > 0.75:
            print(f"\n🎉 EXCELLENT PERFORMANCE! 75%+ accuracy achieved")
        elif accuracy > 0.65:
            print(f"\n👍 GOOD PERFORMANCE! 65%+ accuracy achieved")
        else:
            print(f"\n⚠️  Performance needs improvement") 