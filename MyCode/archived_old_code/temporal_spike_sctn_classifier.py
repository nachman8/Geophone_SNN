#!/usr/bin/env python3
"""
Temporal Spike sctnN Classifier
Uses spike rate/density as temporal signals for sctnN learning
"""

import numpy as np
import pickle
import os
import sys

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import create_SCTN
from sctnN.layers import SCTNLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TemporalSpikeSCTNClassifier:
    """
    sctnN classifier using temporal spike density patterns
    """
    
    def __init__(self, time_window_ms=100, n_key_resonators=6, clk_freq=153600):
        self.time_window_ms = time_window_ms
        self.n_key_resonators = n_key_resonators
        self.clk_freq = clk_freq
        self.network = None
        self.trained = False
        
        # Key resonator frequencies for car detection
        self.key_resonator_indices = [2, 3, 4, 5, 6, 7]  # Car range: 30.5-47.7 Hz
        
        print(f"🧠 Temporal Spike sctnN Classifier:")
        print(f"   ⏱️  Time window: {time_window_ms}ms")
        print(f"   📊 Key resonators: {n_key_resonators}")
        print(f"   🎯 Target: Car pattern detection via spike rates")
    
    def convert_spikes_to_temporal_signal(self, spike_events, duration, clk_freq):
        """
        Convert dense spike timestamps to temporal spike rate signal
        """
        # Create time bins
        time_bins_per_second = 1000 // self.time_window_ms  # bins per second
        n_time_bins = int(duration * time_bins_per_second)
        bin_size_samples = int(clk_freq * self.time_window_ms / 1000)
        
        # Count spikes in each time bin
        spike_rates = np.zeros(n_time_bins)
        
        if len(spike_events) > 0:
            # Convert spike timestamps to bin indices
            bin_indices = spike_events // bin_size_samples
            bin_indices = bin_indices[bin_indices < n_time_bins]  # Keep valid bins
            
            # Count spikes per bin
            for bin_idx in bin_indices:
                spike_rates[bin_idx] += 1
        
        return spike_rates
    
    def extract_temporal_features(self, resonator_outputs, duration):
        """
        Extract temporal features from resonator spike outputs
        """
        temporal_signals = []
        
        # Process key resonators only
        for clk_freq, spikes_arrays in resonator_outputs.items():
            if clk_freq == 153600:  # Main clock frequency
                for i, resonator_idx in enumerate(self.key_resonator_indices):
                    if resonator_idx < len(spikes_arrays):
                        spike_events = spikes_arrays[resonator_idx]
                        
                        # Convert to temporal signal
                        spike_rate_signal = self.convert_spikes_to_temporal_signal(
                            spike_events, duration, clk_freq
                        )
                        
                        temporal_signals.append(spike_rate_signal)
        
        # Combine signals into feature matrix
        if temporal_signals:
            # Stack signals (resonators x time_bins)
            feature_matrix = np.array(temporal_signals)
            
            # Extract temporal features for sctnN
            features = []
            
            # For each resonator
            for resonator_signal in feature_matrix:
                # Basic statistics
                features.extend([
                    np.mean(resonator_signal),
                    np.std(resonator_signal),
                    np.max(resonator_signal),
                    np.sum(resonator_signal > np.mean(resonator_signal) + np.std(resonator_signal))
                ])
            
            # Temporal dynamics across resonators
            if len(feature_matrix) > 1:
                # Cross-resonator correlations (key for car patterns)
                for i in range(len(feature_matrix) - 1):
                    correlation = np.corrcoef(feature_matrix[i], feature_matrix[i+1])[0,1]
                    features.append(correlation if not np.isnan(correlation) else 0.0)
                
                # Temporal variability patterns
                combined_signal = np.mean(feature_matrix, axis=0)
                features.extend([
                    np.var(combined_signal),
                    len(combined_signal[combined_signal > np.percentile(combined_signal, 90)])
                ])
            
            return np.array(features)
        else:
            return np.zeros(32)  # Default feature size
    
    def create_network(self, n_features):
        """Create sctnN network for temporal pattern classification"""
        print(f"🏗️  Creating temporal sctnN network...")
        
        # Create network
        self.network = SpikingNetwork()
        self.network.clk_freq = self.clk_freq
        self.network.add_amplitude(100)  # Lower amplitude for more sensitive response
        
        # Input layer
        input_neurons = []
        for i in range(n_features):
            neuron = create_SCTN()
            neuron.activation_function = 0  # IDENTITY
            neuron.membrane_should_reset = False
            neuron.threshold_pulse = 5  # Lower threshold for sensitivity
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Hidden layer for temporal pattern detection
        hidden_neurons = []
        n_hidden = 16  # Smaller network for better learning
        
        for i in range(n_hidden):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.normal(0.3, 0.1, n_features).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.1, 0.8)
            
            neuron.activation_function = 0  # IDENTITY
            neuron.membrane_should_reset = False
            neuron.theta = -0.5  # Lower threshold
            neuron.leakage_factor = 1
            neuron.leakage_period = 3
            
            # Set STDP for pattern learning
            neuron.set_stdp(
                A_LTP=20e-5,  # Higher learning rate
                A_LTD=-15e-5,
                tau=2e-5,
                clk_freq=self.clk_freq,
                wmax=1.5,
                wmin=0.1
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer (binary classification)
        output_neurons = []
        
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.normal(0.4, 0.1, n_hidden).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.1, 0.8)
            
            neuron.activation_function = 0  # IDENTITY
            neuron.membrane_should_reset = False
            neuron.theta = -0.6
            neuron.leakage_factor = 1
            neuron.leakage_period = 5
            
            # Set STDP for classification
            neuron.set_stdp(
                A_LTP=15e-5,
                A_LTD=-10e-5,
                tau=3e-5,
                clk_freq=self.clk_freq,
                wmax=1.2,
                wmin=0.1
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Set up logging
        for neuron in self.network.neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ Network: {n_features} → {n_hidden} → 2")
        return self.network
    
    def load_and_prepare_data(self, chunks_dir):
        """Load chunk data and prepare temporal features"""
        print(f"📊 Loading and preparing temporal data...")
        
        X = []
        y = []
        
        # Load car chunks
        car_dir = os.path.join(chunks_dir, "car")
        car_index_file = os.path.join(car_dir, "chunk_index.pkl")
        
        if os.path.exists(car_index_file):
            with open(car_index_file, 'rb') as f:
                car_index = pickle.load(f)
            
            print(f"📁 Processing {len(car_index['chunk_files'])} car chunks...")
            for chunk_file in car_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    features = self.extract_temporal_features(
                        chunk_data['resonator_outputs'],
                        chunk_data['duration']
                    )
                    X.append(features)
                    y.append(0)  # Car = 0
        
        # Load car_nothing chunks
        nothing_dir = os.path.join(chunks_dir, "car_nothing")
        nothing_index_file = os.path.join(nothing_dir, "chunk_index.pkl")
        
        if os.path.exists(nothing_index_file):
            with open(nothing_index_file, 'rb') as f:
                nothing_index = pickle.load(f)
            
            print(f"📁 Processing {len(nothing_index['chunk_files'])} car_nothing chunks...")
            for chunk_file in nothing_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    features = self.extract_temporal_features(
                        chunk_data['resonator_outputs'],
                        chunk_data['duration']
                    )
                    X.append(features)
                    y.append(1)  # Nothing = 1
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Dataset prepared:")
        print(f"   Total samples: {len(X)}")
        print(f"   Car samples: {np.sum(y == 0)}")
        print(f"   Nothing samples: {np.sum(y == 1)}")
        print(f"   Feature size: {X.shape[1]}")
        
        return X, y
    
    def train(self, X, y, n_epochs=30):
        """Train the temporal sctnN network"""
        print(f"\n🧠 Training temporal sctnN network...")
        
        if self.network is None:
            self.create_network(X.shape[1])
        
        history = []
        
        for epoch in range(n_epochs):
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X))
            
            for idx in indices:
                sample = X[idx]
                label = y[idx]
                
                # Scale features for better sctnN response
                scaled_sample = sample * 10  # Moderate scaling
                
                # Input to network
                self.network.input_potential(scaled_sample)
                
                # Get output
                output_layer = self.network.layers_neurons[-1]
                car_spikes = len(output_layer.neurons[0].out_spikes())
                nothing_spikes = len(output_layer.neurons[1].out_spikes())
                
                # Prediction based on spike counts
                prediction = 0 if car_spikes > nothing_spikes else 1
                
                if prediction == label:
                    correct_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / len(X)
            history.append(accuracy)
            
            print(f"Epoch {epoch+1}/{n_epochs}: Accuracy = {accuracy:.1%}")
            
            # Early stopping if converged
            if epoch > 5 and len(set(history[-3:])) == 1:
                print(f"✅ Training converged at epoch {epoch+1}")
                break
        
        self.trained = True
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained network"""
        if not self.trained:
            print("❌ Network not trained yet")
            return None
        
        predictions = []
        
        for sample in X_test:
            scaled_sample = sample * 10
            self.network.input_potential(scaled_sample)
            
            output_layer = self.network.layers_neurons[-1]
            car_spikes = len(output_layer.neurons[0].out_spikes())
            nothing_spikes = len(output_layer.neurons[1].out_spikes())
            
            prediction = 0 if car_spikes > nothing_spikes else 1
            predictions.append(prediction)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Car', 'Nothing'])
        cm = confusion_matrix(y_test, predictions)
        
        return accuracy, report, cm

def main():
    """Main function"""
    print("🚀 TEMPORAL SPIKE sctnN CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    classifier = TemporalSpikeSCTNClassifier(time_window_ms=50, n_key_resonators=6)
    
    # Load data
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    X, y = classifier.load_and_prepare_data(chunks_dir)
    
    if len(X) == 0:
        print("❌ No data loaded")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train
    history = classifier.train(X_train, y_train, n_epochs=20)
    
    # Evaluate
    accuracy, report, cm = classifier.evaluate(X_test, y_test)
    
    print(f"\n🎯 TEMPORAL RESULTS:")
    print(f"📊 Test Accuracy: {accuracy:.1%}")
    print(f"\n📈 Classification Report:")
    print(report)
    print(f"\n🔥 Confusion Matrix:")
    print(cm)
    
    # Learning analysis
    if len(history) > 3:
        initial_acc = history[0]
        final_acc = history[-1]
        improvement = final_acc - initial_acc
        
        print(f"\n🧠 LEARNING ANALYSIS:")
        print(f"   Initial: {initial_acc:.1%}")
        print(f"   Final: {final_acc:.1%}")
        print(f"   Improvement: {improvement:+.1%}")
        
        if improvement > 0.1:
            print("   ✅ SIGNIFICANT LEARNING DETECTED!")
        elif improvement > 0.05:
            print("   👍 Moderate learning detected")
        else:
            print("   ❌ Limited learning")
    
    return classifier

if __name__ == "__main__":
    classifier = main() 