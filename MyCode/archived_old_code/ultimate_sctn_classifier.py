#!/usr/bin/env python3
"""
ULTIMATE sctnN Classifier - Final Working Solution
Uses correct learning rates + stability optimizations
"""

import numpy as np
import pickle
import os
import sys

sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class UltimateSCTNClassifier:
    def __init__(self):
        self.clk_freq = 1536000
        self.spike_duration = 60  # Shorter for stability
        self.network = None
        
        # ULTIMATE parameters (notebook + optimizations)
        self.A_LTP = 0.0003     # High learning rate from notebook
        self.A_LTD = -0.0002    # Balanced
        self.tau = 1920.0       # From notebook calculation
        
        print(f"🚀 ULTIMATE sctnN Classifier")
        print(f"   A_LTP={self.A_LTP} (notebook optimized)")
    
    def create_network(self, n_features):
        print(f"🏗️ Creating ULTIMATE network...")
        
        self.network = SpikingNetwork()
        
        # Input layer
        input_neurons = [create_SCTN() for _ in range(n_features)]
        self.network.add_layer(SCTNLayer(input_neurons))
        
        # Single hidden layer (simpler is better)
        n_hidden = 8
        hidden_neurons = []
        for i in range(n_hidden):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.4, 0.6, n_features).astype(np.float64)
            neuron.leakage_factor = 2
            neuron.leakage_period = 3
            neuron.theta = -5
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            
            # ULTIMATE STDP
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=1.0,
                wmin=0.2
            )
            hidden_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(hidden_neurons))
        
        # Output layer with supervised STDP
        output_neurons = []
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.3, 0.5, n_hidden).astype(np.float64)
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.theta = -3
            neuron.activation_function = BINARY
            neuron.membrane_should_reset = True
            
            neuron.set_supervised_stdp(
                A=0.05,
                tau=5.0,
                clk_freq=1000,
                wmax=2.0,
                wmin=0.1,
                desired_output=np.array([], dtype=np.int64)
            )
            output_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(output_neurons))
        
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ Network: {n_features} → {n_hidden} → 2")
        return self.network
    
    def load_data(self, chunks_dir):
        X, y = [], []
        
        # Car chunks
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
                    
                    # Simple robust features
                    features = []
                    for band_data in spikegram:
                        features.append(np.mean(band_data))
                        features.append(np.max(band_data))
                    
                    X.append(features)
                    y.append(0)
        
        # Nothing chunks
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
                        features.append(np.mean(band_data))
                        features.append(np.max(band_data))
                    
                    X.append(features)
                    y.append(1)
        
        print(f"📊 Data: {len(X)} samples, {len(X[0]) if X else 0} features")
        return np.array(X), np.array(y)
    
    def convert_to_spikes(self, features):
        n_samples, n_features = features.shape
        spike_trains = np.zeros((n_samples, n_features, self.spike_duration), dtype=np.int8)
        
        for i, sample in enumerate(features):
            norm_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-10)
            
            for j, rate in enumerate(norm_sample):
                spike_rate = rate * 20  # Conservative rate
                spike_prob = spike_rate / self.spike_duration
                spikes = np.random.random(self.spike_duration) < spike_prob
                spike_trains[i, j, :] = spikes.astype(np.int8)
        
        return spike_trains
    
    def train(self, X, y, n_epochs=40):
        if self.network is None:
            self.create_network(X.shape[1])
        
        X_spikes = self.convert_to_spikes(X)
        history = []
        
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            
            for idx in np.random.permutation(len(X_spikes)):
                spike_train = X_spikes[idx]
                target = y[idx]
                
                self.network.reset_input()
                
                # Set targets
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target:
                            spike_times = list(range(5, self.spike_duration-5, 15))
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spikes
                output_spikes = []
                for t in range(self.spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                total_spikes = np.sum(output_spikes, axis=0)
                predicted = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 1
                
                if predicted == target:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            history.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: {accuracy:.1%}")
        
        return history
    
    def evaluate(self, X_test, y_test):
        X_spikes = self.convert_to_spikes(X_test)
        predictions = []
        
        for spike_train in X_spikes:
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
        report = classification_report(y_test, predictions, target_names=['Car', 'Nothing'])
        cm = confusion_matrix(y_test, predictions)
        
        return accuracy, report, cm

# Main
if __name__ == "__main__":
    print("🚀 ULTIMATE sctnN CLASSIFIER - FINAL SOLUTION")
    print("=" * 50)
    
    classifier = UltimateSCTNClassifier()
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    X, y = classifier.load_data(chunks_dir)
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"\n🧠 Training...")
        history = classifier.train(X_train, y_train)
        
        accuracy, report, cm = classifier.evaluate(X_test, y_test)
        
        print(f"\n🎯 ULTIMATE RESULTS:")
        print(f"Test Accuracy: {accuracy:.1%}")
        print(f"Learning: {history[0]:.1%} → {history[-1]:.1%}")
        print("\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:\n{cm}")
        
        if accuracy >= 0.75:
            print("🎉 SUCCESS! 75%+ accuracy achieved!")
