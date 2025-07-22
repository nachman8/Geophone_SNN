#!/usr/bin/env python3
"""
CORRECTED Labeling sctnN Classifier
FIXED: Nothing=0, Detection=1 (correct labeling logic)
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

class CorrectedLabelingSCTNClassifier:
    def __init__(self):
        self.clk_freq = 1536000
        self.spike_duration = 80
        self.network = None
        
        # Optimized parameters from notebook
        self.A_LTP = 0.0003
        self.A_LTD = -0.0002
        self.tau = 1920.0
        
        print(f"🚀 CORRECTED Labeling sctnN Classifier")
        print(f"   ✅ FIXED: Nothing=0, Detection=1")
        print(f"   📈 A_LTP={self.A_LTP}")
    
    def create_network(self, n_features):
        print(f"🏗️ Creating network...")
        
        self.network = SpikingNetwork()
        
        # Input layer
        input_neurons = [create_SCTN() for _ in range(n_features)]
        self.network.add_layer(SCTNLayer(input_neurons))
        
        # Hidden layer
        n_hidden = 10
        hidden_neurons = []
        for i in range(n_hidden):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.4, 0.6, n_features).astype(np.float64)
            neuron.leakage_factor = 2
            neuron.leakage_period = 4
            neuron.theta = -6
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=1.0,
                wmin=0.1
            )
            hidden_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(hidden_neurons))
        
        # Output layer (binary: nothing=0, detection=1)
        output_neurons = []
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(0.3, 0.5, n_hidden).astype(np.float64)
            neuron.leakage_factor = 1
            neuron.leakage_period = 3
            neuron.theta = -4
            neuron.activation_function = BINARY
            neuron.membrane_should_reset = True
            
            neuron.set_supervised_stdp(
                A=0.04,
                tau=8.0,
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
    
    def detect_car_patterns(self, spikegram):
        """Detect car patterns in spikegram"""
        # Focus on car frequency bands (36.74 Hz - bands 1,2,3)
        car_bands = spikegram[1:4]  # CAR_APPROACH, CAR_PEAK, CAR_TAIL
        
        # Calculate energy in car bands
        car_energy = np.sum([np.sum(band) for band in car_bands])
        total_energy = np.sum([np.sum(band) for band in spikegram])
        
        # Car pattern: high energy in car bands + temporal variability
        car_ratio = car_energy / (total_energy + 1e-10)
        
        # Temporal variability (cars are bursty)
        segment_energies = []
        segment_size = len(spikegram[0]) // 8  # 8 segments
        for i in range(8):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size
            segment_energy = np.sum([np.sum(band[start_idx:end_idx]) for band in car_bands])
            segment_energies.append(segment_energy)
        
        if len(segment_energies) > 1:
            variability = np.std(segment_energies) / (np.mean(segment_energies) + 1e-10)
        else:
            variability = 0
        
        # Car detection criteria
        has_car = (
            (car_ratio > 0.25) and          # Strong car band activity
            (variability > 0.3) and        # Bursty temporal pattern
            (car_energy > 1000)            # Minimum energy threshold
        )
        
        return has_car
    
    def load_data(self, chunks_dir):
        X, y = [], []
        
        print("📊 Loading data with CORRECTED labeling...")
        
        # Load car chunks (signal file)
        car_dir = os.path.join(chunks_dir, "car")
        car_index_file = os.path.join(car_dir, "chunk_index.pkl")
        if os.path.exists(car_index_file):
            with open(car_index_file, 'rb') as f:
                car_index = pickle.load(f)
            
            car_detections = 0
            car_nothing = 0
            
            for chunk_file in car_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    
                    # Extract features
                    features = []
                    for band_data in spikegram:
                        features.extend([
                            np.mean(band_data),
                            np.max(band_data),
                            np.std(band_data)
                        ])
                    
                    # CORRECTED DETECTION LOGIC
                    has_car = self.detect_car_patterns(spikegram)
                    
                    X.append(features)
                    if has_car:
                        y.append(1)  # ✅ CORRECT: Car detected = 1
                        car_detections += 1
                    else:
                        y.append(0)  # ✅ CORRECT: Nothing = 0
                        car_nothing += 1
            
            print(f"   Car chunks: {car_detections} detections, {car_nothing} nothing")
        
        # Load car_nothing chunks (background file)
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
                    
                    features = []
                    for band_data in spikegram:
                        features.extend([
                            np.mean(band_data),
                            np.max(band_data),
                            np.std(band_data)
                        ])
                    
                    X.append(features)
                    y.append(0)  # ✅ CORRECT: Nothing chunks = 0 (always)
                    nothing_count += 1
            
            print(f"   Nothing chunks: {nothing_count} nothing (all)")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 CORRECTED Dataset:")
        print(f"   Total samples: {len(X)}")
        print(f"   Detections (1): {np.sum(y == 1)}")
        print(f"   Nothing (0): {np.sum(y == 0)}")
        
        return X, y
    
    def convert_to_spikes(self, features):
        n_samples, n_features = features.shape
        spike_trains = np.zeros((n_samples, n_features, self.spike_duration), dtype=np.int8)
        
        for i, sample in enumerate(features):
            norm_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-10)
            
            for j, rate in enumerate(norm_sample):
                spike_rate = rate * 25
                spike_prob = spike_rate / self.spike_duration
                spikes = np.random.random(self.spike_duration) < spike_prob
                spike_trains[i, j, :] = spikes.astype(np.int8)
        
        return spike_trains
    
    def train(self, X, y, n_epochs=50):
        if self.network is None:
            self.create_network(X.shape[1])
        
        X_spikes = self.convert_to_spikes(X)
        history = []
        
        print(f"🧠 Training with CORRECTED labels...")
        
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            
            for idx in np.random.permutation(len(X_spikes)):
                spike_train = X_spikes[idx]
                target = y[idx]
                
                self.network.reset_input()
                
                # Set targets with CORRECTED logic
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target:
                            # Target neuron should spike
                            if target == 1:  # Detection
                                spike_times = list(range(8, self.spike_duration-8, 16))
                            else:  # Nothing
                                spike_times = list(range(12, self.spike_duration-12, 20))
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target should not spike
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spikes
                output_spikes = []
                for t in range(self.spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                total_spikes = np.sum(output_spikes, axis=0)
                predicted = np.argmax(total_spikes) if np.sum(total_spikes) > 0 else 0  # Default to nothing
                
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
    print("🚀 CORRECTED LABELING sctnN CLASSIFIER")
    print("=" * 50)
    print("FIXED: Nothing=0, Detection=1")
    print()
    
    classifier = CorrectedLabelingSCTNClassifier()
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    X, y = classifier.load_data(chunks_dir)
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"\n📊 Training on {len(X_train)} samples...")
        history = classifier.train(X_train, y_train, n_epochs=60)
        
        accuracy, report, cm = classifier.evaluate(X_test, y_test)
        
        print(f"\n🎯 CORRECTED RESULTS:")
        print(f"Test Accuracy: {accuracy:.1%}")
        print(f"Learning: {history[0]:.1%} → {history[-1]:.1%}")
        print("\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Analysis
        print(f"\n📈 CORRECTED Analysis:")
        print(f"   Nothing detection: {cm[0,0]}/{cm[0,0]+cm[0,1]} = {cm[0,0]/(cm[0,0]+cm[0,1]):.1%}")
        if cm.shape[0] > 1:
            print(f"   Car detection: {cm[1,1]}/{cm[1,0]+cm[1,1]} = {cm[1,1]/(cm[1,0]+cm[1,1]):.1%}") 