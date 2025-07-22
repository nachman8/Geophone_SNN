#!/usr/bin/env python3
"""
Raw Resonator sctnN Classifier
Uses raw spike events from resonators instead of processed spectrograms for sctnN learning
"""

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import sys

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import create_SCTN
from sctnN.layers import SCTNLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class RawResonatorSCTNClassifier:
    """
    sctnN classifier using raw resonator spike events
    """
    
    def __init__(self, n_input_neurons=30, clk_freq=1536000):
        self.clk_freq = clk_freq
        self.n_input_neurons = n_input_neurons
        self.network = None
        self.trained = False
        
    def create_network(self):
        """Create sctnN network following proper examples"""
        print(f"🧠 Creating sctnN network with {self.n_input_neurons} input neurons")
        
        # Create spiking network (correct way - no parameters)
        self.network = SpikingNetwork()
        self.network.clk_freq = self.clk_freq
        self.network.add_amplitude(1000)
        
        # Input layer
        input_neurons = []
        for i in range(self.n_input_neurons):
            neuron = create_SCTN()
            neuron.activation_function = 0  # IDENTITY
            neuron.membrane_should_reset = False
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Output layer with STDP
        output_neurons = []
        
        # Car classifier neuron
        car_neuron = create_SCTN()
        car_neuron.synapses_weights = np.random.normal(0.5, 0.2, self.n_input_neurons).astype(np.float64)
        car_neuron.synapses_weights = np.clip(car_neuron.synapses_weights, 0.1, 1.0)
        car_neuron.activation_function = 0  # IDENTITY
        car_neuron.membrane_should_reset = False
        car_neuron.theta = -0.8
        
        # Set STDP for car neuron
        car_neuron.set_stdp(
            A_LTP=10e-5, A_LTD=-8e-5, tau=1e-5,
            clk_freq=self.clk_freq, wmax=2.0, wmin=0.1
        )
        output_neurons.append(car_neuron)
        
        # Nothing classifier neuron
        nothing_neuron = create_SCTN()
        nothing_neuron.synapses_weights = np.random.normal(0.5, 0.2, self.n_input_neurons).astype(np.float64)
        nothing_neuron.synapses_weights = np.clip(nothing_neuron.synapses_weights, 0.1, 1.0)
        nothing_neuron.activation_function = 0  # IDENTITY
        nothing_neuron.membrane_should_reset = False
        nothing_neuron.theta = -0.8
        
        # Set STDP for nothing neuron
        nothing_neuron.set_stdp(
            A_LTP=10e-5, A_LTD=-8e-5, tau=1e-5,
            clk_freq=self.clk_freq, wmax=2.0, wmin=0.1
        )
        output_neurons.append(nothing_neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Set up logging
        for neuron in self.network.neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"✅ Network created: {self.n_input_neurons} inputs → 2 outputs with STDP")
        
    def extract_raw_spike_features(self, resonator_outputs, duration, signal_type='car'):
        """
        Extract features from raw resonator spike events
        """
        # Target resonator frequencies for different signal types
        if signal_type == 'car':
            target_freqs = [30.5, 34.7, 37.2, 40.2, 43.6, 47.7]  # Car frequency range
        else:
            target_freqs = [63.6, 69.4, 76.3]  # Human frequency range
            
        spike_features = []
        
        # Process each clock frequency group
        for clk_freq, spikes_arrays in resonator_outputs.items():
            # Get frequencies for this clock
            if clk_freq == 153600:
                freqs = [22.1, 28.8, 30.5, 34.7, 37.2, 40.2, 43.6, 47.7, 52.6, 58.7, 63.6, 69.4, 76.3, 89.8, 95.4]
            else:
                continue  # Skip other clock frequencies for now
                
            for freq, spike_events in zip(freqs, spikes_arrays):
                if freq in target_freqs and len(spike_events) > 0:
                    # Convert spike events to temporal features
                    spike_times = spike_events / clk_freq  # Convert to seconds
                    
                    # Create temporal features
                    if len(spike_times) > 1:
                        # Inter-spike intervals
                        isi = np.diff(spike_times)
                        
                        # Basic temporal features
                        features = [
                            len(spike_times) / duration,  # Spike rate
                            np.mean(isi) if len(isi) > 0 else 0,  # Mean ISI
                            np.std(isi) if len(isi) > 0 else 0,   # ISI variability
                            np.min(isi) if len(isi) > 0 else 0,   # Min ISI
                            np.max(isi) if len(isi) > 0 else 0,   # Max ISI
                        ]
                    else:
                        features = [0, 0, 0, 0, 0]
                    
                    spike_features.extend(features)
        
        # Pad or truncate to fixed size
        target_size = self.n_input_neurons
        if len(spike_features) < target_size:
            spike_features.extend([0] * (target_size - len(spike_features)))
        else:
            spike_features = spike_features[:target_size]
            
        return np.array(spike_features)
    
    def load_chunk_data(self, chunks_base_dir):
        """
        Load raw resonator spike data from saved chunks
        """
        print(f"🔄 Loading raw resonator data from {chunks_base_dir}")
        
        chunk_data = {}
        
        for signal_type in ['car', 'car_nothing']:
            chunk_dir = os.path.join(chunks_base_dir, signal_type)
            index_file = os.path.join(chunk_dir, 'chunk_index.pkl')
            
            if os.path.exists(index_file):
                print(f"📁 Loading {signal_type} chunks...")
                
                with open(index_file, 'rb') as f:
                    chunk_index = pickle.load(f)
                
                chunks = []
                for chunk_file in chunk_index['chunk_files']:
                    if os.path.exists(chunk_file):
                        with open(chunk_file, 'rb') as f:
                            chunk = pickle.load(f)
                        chunks.append(chunk)
                
                chunk_data[signal_type] = chunks
                print(f"   ✅ Loaded {len(chunks)} chunks for {signal_type}")
        
        return chunk_data
    
    def prepare_training_data(self, chunk_data):
        """
        Prepare training data from raw resonator outputs
        """
        print(f"\n🎯 PREPARING TRAINING DATA FROM RAW RESONATOR SPIKES")
        
        X = []
        y = []
        
        # Process car signal chunks
        if 'car' in chunk_data:
            for chunk in chunk_data['car']:
                if 'resonator_outputs' in chunk:
                    features = self.extract_raw_spike_features(
                        chunk['resonator_outputs'], 
                        chunk['duration'], 
                        'car'
                    )
                    X.append(features)
                    y.append(0)  # Car = 0
        
        # Process car nothing chunks
        if 'car_nothing' in chunk_data:
            for chunk in chunk_data['car_nothing']:
                if 'resonator_outputs' in chunk:
                    features = self.extract_raw_spike_features(
                        chunk['resonator_outputs'], 
                        chunk['duration'], 
                        'car'
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
    
    def train(self, X, y, n_epochs=100):
        """
        Train the sctnN network with raw spike features
        """
        print(f"\n🧠 Training sctnN with raw resonator spikes...")
        
        if self.network is None:
            self.create_network()
        
        history = []
        
        for epoch in range(n_epochs):
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X))
            
            for idx in indices:
                sample = X[idx]
                label = y[idx]
                
                # Scale features for sctnN input
                scaled_sample = sample * 100  # Scale for better sctnN response
                
                # Input to network using proper sctnN method
                self.network.input_potential(scaled_sample)
                
                # Get network output
                output_layer = self.network.layers_neurons[-1]
                car_neuron = output_layer.neurons[0]    # Car classifier
                nothing_neuron = output_layer.neurons[1] # Nothing classifier
                
                # Compare spike counts for prediction
                car_spikes = len(car_neuron.out_spikes()) if hasattr(car_neuron, 'out_spikes') else 0
                nothing_spikes = len(nothing_neuron.out_spikes()) if hasattr(nothing_neuron, 'out_spikes') else 0
                
                prediction = 0 if car_spikes > nothing_spikes else 1
                
                if prediction == label:
                    correct_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / len(X)
            history.append(accuracy)
            
            print(f"Epoch {epoch+1}/{n_epochs}: Accuracy = {accuracy:.1%}")
            
            # Early stopping if accuracy is stable
            if epoch > 10 and abs(accuracy - history[-5]) < 0.001:
                print(f"✅ Training converged at epoch {epoch+1}")
                break
        
        self.trained = True
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained network
        """
        if not self.trained:
            print("❌ Network not trained yet")
            return None
        
        predictions = []
        
        for sample in X_test:
            # Scale features for sctnN input
            scaled_sample = sample * 100
            
            # Input to network
            self.network.input_potential(scaled_sample)
            
            # Get prediction
            output_layer = self.network.layers_neurons[-1]
            car_neuron = output_layer.neurons[0]
            nothing_neuron = output_layer.neurons[1]
            
            car_spikes = len(car_neuron.out_spikes()) if hasattr(car_neuron, 'out_spikes') else 0
            nothing_spikes = len(nothing_neuron.out_spikes()) if hasattr(nothing_neuron, 'out_spikes') else 0
            
            prediction = 0 if car_spikes > nothing_spikes else 1
            predictions.append(prediction)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Car', 'Nothing'])
        cm = confusion_matrix(y_test, predictions)
        
        return accuracy, report, cm

def main():
    """
    Main function to train and test raw resonator sctnN classifier
    """
    print("🚀 RAW RESONATOR sctnN CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    classifier = RawResonatorSCTNClassifier(n_input_neurons=30)
    
    # Load chunk data
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    chunk_data = classifier.load_chunk_data(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return
    
    # Prepare training data
    X, y = classifier.prepare_training_data(chunk_data)
    
    if len(X) == 0:
        print("❌ No training data prepared")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train
    history = classifier.train(X_train, y_train, n_epochs=50)
    
    # Evaluate
    print(f"\n🎯 EVALUATION:")
    accuracy, report, cm = classifier.evaluate(X_test, y_test)
    
    print(f"📊 Test Accuracy: {accuracy:.1%}")
    print(f"\n📈 Classification Report:")
    print(report)
    print(f"\n🔥 Confusion Matrix:")
    print(cm)
    
    # Check if real learning happened
    if len(history) > 5:
        final_acc = history[-1]
        initial_acc = history[0]
        improvement = final_acc - initial_acc
        
        print(f"\n🧠 LEARNING ANALYSIS:")
        print(f"   Initial accuracy: {initial_acc:.1%}")
        print(f"   Final accuracy: {final_acc:.1%}")
        print(f"   Improvement: {improvement:.1%}")
        
        if improvement > 0.05:
            print("   ✅ REAL LEARNING DETECTED!")
        else:
            print("   ❌ No significant learning")
    
    return classifier

if __name__ == "__main__":
    classifier = main() 