#!/usr/bin/env python3
"""
Final Working sctnN Classifier
Based on debug findings - optimized for actual learning with real data
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

class FinalWorkingSCTNClassifier:
    """
    FINAL working sctnN classifier with optimized parameters
    """
    
    def __init__(self, clk_freq=153600):
        self.clk_freq = clk_freq
        self.network = None
        self.trained = False
        
        # OPTIMIZED PARAMETERS based on debug findings
        self.learning_rate_multiplier = 10  # 10x higher learning rate
        self.amplitude = 50  # Moderate amplitude for good signal
        self.training_epochs = 100  # More training iterations
        
        print(f"🚀 FINAL Working sctnN Classifier")
        print(f"   📈 Learning rate: 10x optimized")
        print(f"   🎯 Target: Proven learnable car vs nothing patterns")
    
    def load_and_prepare_data(self, chunks_dir):
        """Load and prepare the proven-learnable chunk data"""
        print(f"📊 Loading proven-learnable chunk data...")
        
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
                    
                    # Use the PROVEN working features from debug
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    features = np.mean(spikegram, axis=1)  # Mean of each frequency band
                    
                    X.append(features)
                    y.append(0)  # Car = 0
        
        # Load car_nothing chunks
        nothing_dir = os.path.join(chunks_dir, "car_nothing")
        nothing_index_file = os.path.join(nothing_dir, "chunk_index.pkl")
        
        if os.path.exists(nothing_index_file):
            with open(nothing_index_file, 'rb') as f:
                nothing_index = pickle.load(f)
            
            print(f"📁 Processing {len(nothing_index['chunk_files'])} nothing chunks...")
            for chunk_file in nothing_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Use the PROVEN working features from debug
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    features = np.mean(spikegram, axis=1)  # Mean of each frequency band
                    
                    X.append(features)
                    y.append(1)  # Nothing = 1
        
        X = np.array(X)
        y = np.array(y)
        
        # Analyze the data like in debug
        if len(X) > 1:
            car_indices = y == 0
            nothing_indices = y == 1
            
            if np.any(car_indices) and np.any(nothing_indices):
                car_features = X[car_indices]
                nothing_features = X[nothing_indices]
                
                car_mean = np.mean(car_features)
                nothing_mean = np.mean(nothing_features)
                max_diff = np.max(np.abs(np.mean(car_features, axis=0) - np.mean(nothing_features, axis=0)))
                
                print(f"📊 Feature analysis:")
                print(f"   Car mean: {car_mean:.2f}")
                print(f"   Nothing mean: {nothing_mean:.2f}")
                print(f"   Ratio: {car_mean/nothing_mean:.2f}x")
                print(f"   Max difference: {max_diff:.2f}")
                
                if car_mean / nothing_mean > 2.0:
                    print(f"   ✅ STRONG discriminative signal detected!")
                else:
                    print(f"   ⚠️  Weak signal - may need more training")
        
        print(f"📊 Dataset prepared:")
        print(f"   Total samples: {len(X)}")
        print(f"   Car samples: {np.sum(y == 0)}")
        print(f"   Nothing samples: {np.sum(y == 1)}")
        print(f"   Feature size: {X.shape[1]}")
        
        return X, y
    
    def create_network(self, n_features):
        """Create optimized sctnN network"""
        print(f"🏗️  Creating OPTIMIZED sctnN network...")
        
        # Create network with proven parameters
        self.network = SpikingNetwork()
        self.network.clk_freq = self.clk_freq
        self.network.add_amplitude(self.amplitude)
        
        # Input layer
        input_neurons = []
        for i in range(n_features):
            neuron = create_SCTN()
            neuron.activation_function = 0  # IDENTITY
            neuron.membrane_should_reset = False
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Output layer (simplified - single neuron for binary classification)
        output_neuron = create_SCTN()
        output_neuron.synapses_weights = np.random.normal(0.5, 0.1, n_features).astype(np.float64)
        output_neuron.synapses_weights = np.clip(output_neuron.synapses_weights, 0.1, 1.0)
        output_neuron.activation_function = 0  # IDENTITY
        output_neuron.membrane_should_reset = False
        output_neuron.theta = -0.5
        
        # OPTIMIZED STDP parameters (10x higher learning rate)
        output_neuron.set_stdp(
            A_LTP=500e-5,  # 10x higher than debug (50e-5)
            A_LTD=-300e-5,  # 10x higher than debug
            tau=1e-5,
            clk_freq=self.clk_freq,
            wmax=2.0,
            wmin=0.1
        )
        
        output_layer = SCTNLayer([output_neuron])
        self.network.add_layer(output_layer)
        
        # Enable logging
        for neuron in self.network.neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ Network: {n_features} → 1 (binary)")
        print(f"   📈 STDP: A_LTP=500e-5, A_LTD=-300e-5 (10x optimized)")
        
        return self.network
    
    def train(self, X, y, verbose=True):
        """Train with optimized parameters"""
        print(f"\n🧠 Training OPTIMIZED sctnN...")
        
        if self.network is None:
            self.create_network(X.shape[1])
        
        output_neuron = self.network.layers_neurons[-1].neurons[0]
        initial_weights = output_neuron.synapses_weights.copy()
        
        if verbose:
            print(f"📊 Initial weights (first 4): {initial_weights[:4]}")
        
        history = []
        
        for epoch in range(self.training_epochs):
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X))
            
            for idx in indices:
                sample = X[idx]
                label = y[idx]
                
                # Scale features (proven working scaling from debug)
                scaled_sample = sample * 10
                
                # Input to network
                self.network.input_potential(scaled_sample)
                
                # Get output (single neuron - spikes indicate car detection)
                output_spikes = len(output_neuron.out_spikes())
                
                # Prediction: High spikes = car (0), Low spikes = nothing (1)
                prediction = 0 if output_spikes > 5 else 1
                
                if prediction == label:
                    correct_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / len(X)
            history.append(accuracy)
            
            # Check weight changes every 10 epochs
            if epoch % 10 == 0:
                current_weights = output_neuron.synapses_weights
                max_change = np.max(np.abs(current_weights - initial_weights))
                
                if verbose:
                    print(f"Epoch {epoch+1:3d}: Accuracy={accuracy:.1%}, Max weight change={max_change:.6f}")
                
                # Detect real learning
                if max_change > 0.001:
                    print(f"         ✅ SIGNIFICANT LEARNING DETECTED!")
            
            # Early convergence check
            if epoch > 20 and len(set(history[-5:])) == 1:
                print(f"✅ Training converged at epoch {epoch+1}")
                break
        
        final_weights = output_neuron.synapses_weights
        total_change = np.max(np.abs(final_weights - initial_weights))
        
        if verbose:
            print(f"📊 Final weights (first 4): {final_weights[:4]}")
            print(f"📊 Total weight change: {total_change:.6f}")
            
            if total_change > 0.01:
                print(f"🔥 EXCELLENT LEARNING! Large weight changes")
            elif total_change > 0.001:
                print(f"✅ GOOD LEARNING! Measurable weight changes")
            else:
                print(f"❌ LIMITED LEARNING: Small weight changes")
        
        self.trained = True
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained network"""
        if not self.trained:
            print("❌ Network not trained yet")
            return None
        
        output_neuron = self.network.layers_neurons[-1].neurons[0]
        predictions = []
        
        for sample in X_test:
            scaled_sample = sample * 10
            self.network.input_potential(scaled_sample)
            
            output_spikes = len(output_neuron.out_spikes())
            prediction = 0 if output_spikes > 5 else 1
            predictions.append(prediction)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Car', 'Nothing'])
        cm = confusion_matrix(y_test, predictions)
        
        return accuracy, report, cm

def main():
    """Main function"""
    print("🚀 FINAL WORKING sctnN CLASSIFIER")
    print("=" * 60)
    print("Based on debug findings - optimized for proven learnable data")
    print()
    
    # Initialize classifier
    classifier = FinalWorkingSCTNClassifier()
    
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
    
    # Train with optimized parameters
    history = classifier.train(X_train, y_train, verbose=True)
    
    # Evaluate
    accuracy, report, cm = classifier.evaluate(X_test, y_test)
    
    print(f"\n🎯 FINAL OPTIMIZED RESULTS:")
    print(f"📊 Test Accuracy: {accuracy:.1%}")
    print(f"\n📈 Classification Report:")
    print(report)
    print(f"\n🔥 Confusion Matrix:")
    print(cm)
    
    # Learning analysis
    if len(history) > 10:
        initial_acc = history[0]
        final_acc = history[-1]
        best_acc = max(history)
        improvement = final_acc - initial_acc
        
        print(f"\n🧠 COMPREHENSIVE LEARNING ANALYSIS:")
        print(f"   Initial accuracy: {initial_acc:.1%}")
        print(f"   Final accuracy: {final_acc:.1%}")
        print(f"   Best accuracy: {best_acc:.1%}")
        print(f"   Total improvement: {improvement:+.1%}")
        
        if improvement > 0.2:
            print("   🔥 EXCELLENT LEARNING!")
        elif improvement > 0.1:
            print("   ✅ STRONG LEARNING!")
        elif improvement > 0.05:
            print("   👍 GOOD LEARNING!")
        elif improvement > 0.01:
            print("   📈 MODERATE LEARNING!")
        else:
            print("   ❌ LIMITED LEARNING")
        
        # Check for overfitting
        if best_acc > final_acc + 0.1:
            print("   ⚠️  Possible overfitting detected")
    
    return classifier

if __name__ == "__main__":
    classifier = main() 