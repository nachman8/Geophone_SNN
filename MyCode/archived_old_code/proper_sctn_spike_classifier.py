#!/usr/bin/env python3
"""
Proper sctnN Spike Classifier - Based on notebook examples
PROPER learning rate: A_LTP=0.00025 (from notebook examples)
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
from sklearn.metrics import accuracy_score, classification_report

class ProperSCTNClassifier:
    def __init__(self, spike_duration=100):
        self.clk_freq = 1536000
        self.spike_duration = spike_duration
        self.network = None
        self.trained = False
        
        # PROPER parameters from notebook examples
        self.time_to_learn = 2.5e-3
        self.A_LTP = 0.00025    # FROM NOTEBOOK: 25x higher!
        self.A_LTD = -0.00015   # FROM NOTEBOOK
        self.tau = self.clk_freq * self.time_to_learn / 2
        
        print(f"🚀 PROPER sctnN Classifier (A_LTP={self.A_LTP})")
    
    def create_network(self, n_features):
        print(f"🏗️  Creating PROPER sctnN network...")
        
        self.network = SpikingNetwork()
        
        # Input layer
        input_neurons = [create_SCTN() for _ in range(n_features)]
        for neuron in input_neurons:
            neuron.activation_function = IDENTITY
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Hidden layer with PROPER STDP
        n_hidden = 15
        hidden_neurons = []
        for i in range(n_hidden):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.3, 0.7, n_features).astype(np.float64)
            neuron.leakage_factor = 2
            neuron.leakage_period = 5
            neuron.theta = -10
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            
            # PROPER STDP from notebook
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=1.0,
                wmin=0.0
            )
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer with supervised STDP
        output_neurons = []
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.1, 0.6, n_hidden).astype(np.float64)
            neuron.leakage_factor = 2
            neuron.leakage_period = 5
            neuron.theta = -10
            neuron.activation_function = BINARY
            neuron.membrane_should_reset = True
            
            # Supervised STDP
            neuron.set_supervised_stdp(
                A=0.03,
                tau=10.0,
                clk_freq=1000,
                wmax=3.0,
                wmin=0.0,
                desired_output=np.array([], dtype=np.int64)
            )
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable logging
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ Network: {n_features} → {n_hidden} → 2")
        return self.network
    
    def convert_to_spikes(self, features):
        n_samples, n_features = features.shape
        spike_trains = np.zeros((n_samples, n_features, self.spike_duration), dtype=np.int8)
        
        for i, sample in enumerate(features):
            # Normalize to [0, 1]
            normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-10)
            
            for j, rate in enumerate(normalized):
                # Convert to spike rate
                spike_rate = rate * 30  # Max 30 Hz
                spike_prob = spike_rate / self.spike_duration
                spikes = np.random.random(self.spike_duration) < spike_prob
                spike_trains[i, j, :] = spikes.astype(np.int8)
        
        return spike_trains
    
    def load_data(self, chunks_dir):
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
                    features = [np.mean(band) for band in spikegram]  # Simple features
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
                    features = [np.mean(band) for band in spikegram]
                    X.append(features)
                    y.append(1)  # Nothing = 1
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Data loaded: {len(X)} samples, {X.shape[1]} features")
        if len(X) > 1:
            car_mean = np.mean(X[y == 0])
            nothing_mean = np.mean(X[y == 1])
            print(f"   Car mean: {car_mean:.2f}, Nothing mean: {nothing_mean:.2f}")
        
        return X, y
    
    def train(self, X, y, n_epochs=50):
        if self.network is None:
            self.create_network(X.shape[1])
        
        print("Converting to spike trains...")
        X_spikes = self.convert_to_spikes(X)
        
        history = []
        
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            
            indices = np.random.permutation(len(X_spikes))
            
            for idx in indices:
                spike_train = X_spikes[idx]
                target = y[idx]
                
                # Reset network
                self.network.reset_input()
                
                # Set supervised targets
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target:
                            # Target neuron should spike
                            spike_times = list(range(10, self.spike_duration-10, 20))
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target should not spike
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spike train (PROPER method)
                output_spikes = []
                for t in range(self.spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)  # PROPER!
                    output_spikes.append(output)
                
                # Prediction
                total_spikes = np.sum(output_spikes, axis=0)
                predicted = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 1
                
                if predicted == target:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            history.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Accuracy = {accuracy:.1%}")
        
        self.trained = True
        return history
    
    def evaluate(self, X_test, y_test):
        X_test_spikes = self.convert_to_spikes(X_test)
        predictions = []
        
        for spike_train in X_test_spikes:
            self.network.reset_input()
            
            output_spikes = []
            for t in range(self.spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            total_spikes = np.sum(output_spikes, axis=0)
            prediction = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 1
            predictions.append(prediction)
        
        accuracy = accuracy_score(y_test, predictions)
        return accuracy, predictions

# Main execution
if __name__ == "__main__":
    print("🚀 PROPER sctnN CLASSIFIER")
    print("Using CORRECT learning rate from notebook examples")
    
    classifier = ProperSCTNClassifier()
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    X, y = classifier.load_data(chunks_dir)
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"\n📊 Training on {len(X_train)} samples...")
        history = classifier.train(X_train, y_train, n_epochs=60)
        
        accuracy, predictions = classifier.evaluate(X_test, y_test)
        
        print(f"\n🎯 RESULTS:")
        print(f"   Test Accuracy: {accuracy:.1%}")
        print(f"   Learning improvement: {history[-1] - history[0]:+.1%}")
        
        if history[-1] - history[0] > 0.1:
            print("   ✅ SIGNIFICANT LEARNING ACHIEVED!")
        else:
            print("   ⚠️  Limited learning detected")
