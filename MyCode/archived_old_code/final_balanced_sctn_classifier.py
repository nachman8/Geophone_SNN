#!/usr/bin/env python3
"""
FINAL BALANCED sctnN Classifier
Corrected labeling + balanced training for stability
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

class FinalBalancedSCTNClassifier:
    def __init__(self):
        self.clk_freq = 1536000
        self.spike_duration = 60  # Shorter for stability
        self.network = None
        
        # Conservative parameters for stability
        self.A_LTP = 0.0002     # Lower for stability
        self.A_LTD = -0.00015   # Balanced
        self.tau = 1920.0
        
        print(f"🚀 FINAL BALANCED sctnN Classifier")
        print(f"   ✅ Corrected: Nothing=0, Detection=1")
        print(f"   🎯 Balanced training for stability")
    
    def create_network(self, n_features):
        print(f"🏗️ Creating balanced network...")
        
        self.network = SpikingNetwork()
        
        # Simpler input layer
        input_neurons = [create_SCTN() for _ in range(n_features)]
        self.network.add_layer(SCTNLayer(input_neurons))
        
        # Smaller hidden layer for stability
        n_hidden = 6
        hidden_neurons = []
        for i in range(n_hidden):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.45, 0.55, n_features).astype(np.float64)
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.theta = -3
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=0.8,
                wmin=0.2
            )
            hidden_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(hidden_neurons))
        
        # Balanced output layer
        output_neurons = []
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.4, 0.6, n_hidden).astype(np.float64)
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.theta = -2
            neuron.activation_function = BINARY
            neuron.membrane_should_reset = True
            
            neuron.set_supervised_stdp(
                A=0.02,  # Lower for stability
                tau=12.0,
                clk_freq=1000,
                wmax=1.5,
                wmin=0.2,
                desired_output=np.array([], dtype=np.int64)
            )
            output_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(output_neurons))
        
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ Balanced Network: {n_features} → {n_hidden} → 2")
        return self.network
    
    def detect_patterns(self, spikegram):
        """Simple but effective pattern detection"""
        # Car bands (focus on 36.74 Hz region)
        car_bands = spikegram[1:4]
        car_energy = np.sum([np.sum(band) for band in car_bands])
        
        # Simple thresholds based on analysis
        has_pattern = car_energy > 5000  # Conservative threshold
        
        return has_pattern
    
    def load_data(self, chunks_dir):
        X, y = [], []
        
        print("📊 Loading with BALANCED labeling...")
        
        # Car chunks
        car_dir = os.path.join(chunks_dir, "car")
        car_index_file = os.path.join(car_dir, "chunk_index.pkl")
        if os.path.exists(car_index_file):
            with open(car_index_file, 'rb') as f:
                car_index = pickle.load(f)
            
            detections = 0
            nothing = 0
            
            for chunk_file in car_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    
                    # Simple features
                    features = [np.mean(band) for band in spikegram]
                    
                    # Pattern detection
                    has_pattern = self.detect_patterns(spikegram)
                    
                    X.append(features)
                    if has_pattern:
                        y.append(1)  # Detection = 1
                        detections += 1
                    else:
                        y.append(0)  # Nothing = 0
                        nothing += 1
            
            print(f"   Car chunks: {detections} detections, {nothing} nothing")
        
        # Nothing chunks (always 0)
        nothing_dir = os.path.join(chunks_dir, "car_nothing")
        nothing_index_file = os.path.join(nothing_dir, "chunk_index.pkl")
        if os.path.exists(nothing_index_file):
            with open(nothing_index_file, 'rb') as f:
                nothing_index = pickle.load(f)
            
            nothing_count = 0
            
            for chunk_file in nothing_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    features = [np.mean(band) for band in spikegram]
                    
                    X.append(features)
                    y.append(0)  # Nothing = 0 (always)
                    nothing_count += 1
            
            print(f"   Nothing chunks: {nothing_count} nothing (all)")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 BALANCED Dataset:")
        print(f"   Total: {len(X)}, Detections: {np.sum(y == 1)}, Nothing: {np.sum(y == 0)}")
        
        return X, y
    
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
        
        print(f"🧠 BALANCED training...")
        
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            
            # Balanced sampling (equal nothing and detection)
            detection_indices = np.where(y == 1)[0]
            nothing_indices = np.where(y == 0)[0]
            
            # Balance the training
            n_samples = min(len(detection_indices), len(nothing_indices))
            if n_samples > 0:
                selected_detection = np.random.choice(detection_indices, n_samples, replace=True)
                selected_nothing = np.random.choice(nothing_indices, n_samples, replace=True)
                epoch_indices = np.concatenate([selected_detection, selected_nothing])
            else:
                epoch_indices = np.random.permutation(len(X_spikes))
            
            for idx in epoch_indices:
                spike_train = X_spikes[idx]
                target = y[idx]
                
                self.network.reset_input()
                
                # Balanced target setting
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target:
                            if target == 1:  # Detection
                                spike_times = list(range(6, self.spike_duration-6, 12))
                            else:  # Nothing
                                spike_times = list(range(10, self.spike_duration-10, 15))
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
                predicted = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 0
                
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
            prediction = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 0
            predictions.append(prediction)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Nothing', 'Detection'])
        cm = confusion_matrix(y_test, predictions)
        
        return accuracy, report, cm

# Main
if __name__ == "__main__":
    print("🚀 FINAL BALANCED sctnN CLASSIFIER")
    print("=" * 50)
    print("Corrected labeling + Balanced training")
    print()
    
    classifier = FinalBalancedSCTNClassifier()
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    X, y = classifier.load_data(chunks_dir)
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"\n📊 Training...")
        history = classifier.train(X_train, y_train)
        
        accuracy, report, cm = classifier.evaluate(X_test, y_test)
        
        print(f"\n🎯 FINAL BALANCED RESULTS:")
        print(f"Test Accuracy: {accuracy:.1%}")
        print(f"Learning: {history[0]:.1%} → {history[-1]:.1%}")
        print(f"Improvement: {history[-1] - history[0]:+.1%}")
        print("\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        if accuracy >= 0.7:
            print("🎉 EXCELLENT! 70%+ accuracy with correct labeling!")
        elif accuracy >= 0.6:
            print("✅ GOOD! 60%+ accuracy with correct labeling!")
