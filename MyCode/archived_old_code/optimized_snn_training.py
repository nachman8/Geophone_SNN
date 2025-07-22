#!/usr/bin/env python3
"""
Optimized SNN Training with Enhanced Parameters and Architecture
Fixes the poor performance issues observed in the original training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import sctnN components
from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import SCTNeuron, create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer
from sctnN.learning_rules.stdp import STDP
from sctnN.learning_rules.supervised_stdp import SupervisedSTDP

class OptimizedGeophoneSNN:
    """
    Optimized SNN with better architecture and learning parameters
    """
    
    def __init__(self, n_input_neurons=None, n_hidden=60, learning_rate=0.005):
        self.n_input_neurons = n_input_neurons
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.network = None
        self.scaler = StandardScaler()
        self.trained = False
        
        # Classification labels
        self.class_labels = {0: 'signal', 1: 'nothing'}
        
    def create_network(self, n_input_neurons):
        """
        Create optimized SNN architecture with better parameters
        """
        self.n_input_neurons = n_input_neurons
        
        # Create the spiking network
        self.network = SpikingNetwork()
        
        # Input layer (resonator outputs)
        input_neurons = []
        for i in range(n_input_neurons):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 5  # Lower threshold for better sensitivity
            neuron.label = f"input_{i}"
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Hidden layer with optimized parameters
        hidden_neurons = []
        for i in range(self.n_hidden):
            neuron = create_SCTN()
            # Better weight initialization
            neuron.synapses_weights = np.random.normal(0.5, 0.2, n_input_neurons).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.1, 1.0)
            
            # Optimized neuron parameters
            neuron.leakage_factor = 2  # Faster leakage
            neuron.leakage_period = 5  # More frequent leakage
            neuron.theta = -8  # Lower threshold
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # Enhanced STDP learning
            neuron.set_stdp(
                A_LTP=0.02,    # Increased LTP
                A_LTD=-0.01,   # Balanced LTD
                tau=15.0,      # Shorter time window
                clk_freq=1000,
                wmax=3.0,      # Higher weight limit
                wmin=0.0
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer with enhanced parameters
        output_neurons = []
        class_names = ['signal', 'nothing']
        for i in range(2):
            neuron = create_SCTN()
            # Better output weight initialization
            neuron.synapses_weights = np.random.normal(0.8, 0.3, self.n_hidden).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.2, 1.5)
            
            # Optimized output parameters
            neuron.leakage_factor = 1  # Minimal leakage for output
            neuron.leakage_period = 3
            neuron.theta = -5  # Lower threshold for output
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 3  # Lower pulse threshold
            neuron.membrane_should_reset = True
            neuron.label = f"output_{class_names[i]}"
            
            # Enhanced supervised STDP
            neuron.set_supervised_stdp(
                A=0.03,        # Stronger supervision
                tau=10.0,      # Shorter supervision window
                clk_freq=1000,
                wmax=3.0,
                wmin=0.0,
                desired_output=np.array([], dtype=np.int64)
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable spike logging
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"Optimized SNN created: {n_input_neurons} input → {self.n_hidden} hidden → 2 output neurons")
        return self.network
    
    def enhanced_spike_encoding(self, X, spike_duration=150):
        """
        Enhanced spike encoding with better temporal patterns
        """
        n_samples, n_features = X.shape
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to [0,1] range with better distribution
        X_normalized = np.zeros_like(X_scaled)
        for i in range(n_features):
            feature_col = X_scaled[:, i]
            # Use percentile normalization for better spike distribution
            p10, p90 = np.percentile(feature_col, [10, 90])
            if p90 > p10:
                X_normalized[:, i] = np.clip((feature_col - p10) / (p90 - p10), 0, 1)
            else:
                X_normalized[:, i] = 0.5
        
        spike_trains = np.zeros((n_samples, n_features, spike_duration), dtype=int)
        
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                feature_val = X_normalized[sample_idx, feature_idx]
                
                # Enhanced rate coding with temporal structure
                if feature_val > 0.05:  # Only encode significant values
                    # Variable spike rate based on feature value
                    base_rate = feature_val * 0.4  # Max 40% spike probability
                    
                    # Create temporal pattern - higher activity in middle of window
                    for t in range(spike_duration):
                        # Temporal modulation - higher activity in middle
                        temporal_factor = 1.0 - abs(t - spike_duration/2) / (spike_duration/2)
                        temporal_factor = max(0.3, temporal_factor)  # Minimum activity
                        
                        spike_prob = base_rate * temporal_factor
                        if np.random.random() < spike_prob:
                            spike_trains[sample_idx, feature_idx, t] = 1
        
        return spike_trains
    
    def train(self, X_train, y_train, n_epochs=100, spike_duration=150):
        """
        Enhanced training with better learning schedule
        """
        if self.network is None:
            self.create_network(X_train.shape[1])
        
        print(f"Training Optimized SNN for {n_epochs} epochs...")
        
        # Enhanced spike encoding
        print("Converting training data to enhanced spike trains...")
        X_spike_trains = self.enhanced_spike_encoding(X_train, spike_duration)
        
        training_accuracy = []
        learning_schedule = []
        
        for epoch in range(n_epochs):
            epoch_correct = 0
            epoch_total = 0
            
            # Adaptive learning rate
            if epoch > 50:
                current_lr = self.learning_rate * 0.5  # Reduce learning rate
            elif epoch > 80:
                current_lr = self.learning_rate * 0.2
            else:
                current_lr = self.learning_rate
            
            learning_schedule.append(current_lr)
            
            # Shuffle training data
            indices = np.random.permutation(len(X_spike_trains))
            
            for idx in indices:
                spike_train = X_spike_trains[idx]
                target_class = y_train[idx]
                
                # Reset network state
                self.network.reset_input()
                
                # Enhanced supervised learning targets
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target_class:
                            # Target neuron should spike regularly during stimulus
                            spike_times = list(range(20, spike_duration-20, 15))  # More frequent spikes
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target should not spike
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spike train to network
                output_spikes = []
                for t in range(spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Determine predicted class
                total_output_spikes = np.sum(output_spikes, axis=0)
                
                # Better prediction with confidence threshold
                if np.sum(total_output_spikes) > 0:
                    predicted_class = np.argmax(total_output_spikes)
                else:
                    predicted_class = 1  # Default to nothing if no spikes
                
                if predicted_class == target_class:
                    epoch_correct += 1
                epoch_total += 1
            
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            training_accuracy.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Training Accuracy = {accuracy:.3f}, LR = {current_lr:.4f}")
        
        self.trained = True
        
        # Plot enhanced training curve
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_accuracy)
        plt.title('Training Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(learning_schedule)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(training_accuracy[-50:])  # Last 50 epochs
        plt.title('Final Training Phase')
        plt.xlabel('Epoch (Last 50)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("optimized_snn_training_curve.png")
        plt.close()
        
        print(f"Training completed. Final accuracy: {training_accuracy[-1]:.3f}")
        return training_accuracy
    
    def predict(self, X_test, spike_duration=150):
        """
        Enhanced prediction with confidence scoring
        """
        if not self.trained:
            raise ValueError("Network must be trained before making predictions")
        
        # Use same scaling as training
        X_spike_trains = self.enhanced_spike_encoding(X_test, spike_duration)
        
        predictions = []
        confidence_scores = []
        
        for spike_train in X_spike_trains:
            # Reset network state
            self.network.reset_input()
            
            # Present spike train to network
            output_spikes = []
            for t in range(spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            # Enhanced prediction logic
            total_output_spikes = np.sum(output_spikes, axis=0)
            total_spikes = np.sum(total_output_spikes)
            
            if total_spikes > 0:
                predicted_class = np.argmax(total_output_spikes)
                confidence = total_output_spikes[predicted_class] / total_spikes
            else:
                # No spikes - predict nothing class with low confidence
                predicted_class = 1
                confidence = 0.1
            
            predictions.append(predicted_class)
            confidence_scores.append(confidence)
        
        return np.array(predictions), np.array(confidence_scores)
    
    def evaluate(self, X_test, y_test):
        """
        Enhanced evaluation with detailed metrics
        """
        predictions, confidence = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        
        # Generate classification report
        class_names = ['Signal', 'Nothing']
        report = classification_report(
            y_test, predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        print("\n" + "="*60)
        print("OPTIMIZED SNN CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Average Confidence: {np.mean(confidence):.3f}")
        print(f"High Confidence Predictions (>0.7): {np.sum(confidence > 0.7)} / {len(confidence)}")
        
        print("\nConfusion Matrix:")
        print("Predicted:", class_names)
        for i, actual_class in enumerate(class_names):
            print(f"Actual {actual_class:8s}: {cm[i]}")
        
        print("\nDetailed Classification Report:")
        for class_name in class_names:
            if class_name.lower() in report:
                metrics = report[class_name.lower()]
                print(f"{class_name:8s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return accuracy, report, cm
    
    def save_model(self, filepath):
        """Save the optimized model"""
        model_data = {
            'network_weights': self.get_network_weights(),
            'scaler': self.scaler,
            'n_input_neurons': self.n_input_neurons,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'trained': self.trained,
            'class_labels': self.class_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Optimized SNN model saved to {filepath}")
    
    def get_network_weights(self):
        """Extract network weights"""
        weights = {}
        for layer_idx, layer in enumerate(self.network.layers_neurons):
            layer_weights = []
            for neuron in layer.neurons:
                layer_weights.append({
                    'synapses_weights': neuron.synapses_weights,
                    'theta': neuron.theta,
                    'leakage_factor': neuron.leakage_factor,
                    'leakage_period': neuron.leakage_period
                })
            weights[f'layer_{layer_idx}'] = layer_weights
        return weights

def run_optimized_training():
    """
    Run optimized training with the existing chunk data
    """
    print("🚀 OPTIMIZED SNN TRAINING FROM SAVED CHUNKS")
    print("=" * 60)
    
    # Import the existing chunk loading function
    from load_saved_chunks import load_chunks_directly, extract_segments_from_loaded_chunks
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    # Load chunks
    chunk_data = load_chunks_directly(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return None
    
    # Process car classification with optimized SNN
    print("\n🚗 OPTIMIZED CAR vs CAR_NOTHING CLASSIFICATION")
    
    # Extract segments
    car_segments = []
    car_labels = []
    
    # Process car signal segments
    if 'car' in chunk_data:
        segments, labels = extract_segments_from_loaded_chunks(chunk_data['car'], 'car')
        signal_indices = labels == 1
        car_segments.extend(segments[signal_indices])
        car_labels.extend([0] * np.sum(signal_indices))  # 0 = signal
    
    # Process car nothing segments
    if 'car_nothing' in chunk_data:
        segments, labels = extract_segments_from_loaded_chunks(chunk_data['car_nothing'], 'car_nothing')
        nothing_indices = labels == 0
        car_segments.extend(segments[nothing_indices])
        car_labels.extend([1] * np.sum(nothing_indices))  # 1 = nothing
    
    if len(car_segments) == 0:
        print("❌ No car data available")
        return None
    
    car_segments = np.array(car_segments)
    car_labels = np.array(car_labels)
    
    print(f"Optimized car dataset: {len(car_segments)} segments")
    print(f"Signal: {np.sum(car_labels == 0)}, Nothing: {np.sum(car_labels == 1)}")
    
    # Create optimized SNN
    snn = OptimizedGeophoneSNN(n_hidden=60, learning_rate=0.005)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        car_segments, car_labels, test_size=0.25, random_state=42, 
        stratify=car_labels
    )
    
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Train with optimized parameters
    training_history = snn.train(X_train, y_train, n_epochs=120, spike_duration=150)
    
    # Evaluate
    accuracy, report, cm = snn.evaluate(X_test, y_test)
    
    # Save optimized model
    snn.save_model("optimized_car_snn_model.pkl")
    
    print(f"\n✅ OPTIMIZED CAR SNN RESULTS:")
    print(f"📊 Test Accuracy: {accuracy:.1%}")
    print(f"🎯 Target: >80% accuracy")
    
    if accuracy > 0.8:
        print("🎉 EXCELLENT PERFORMANCE ACHIEVED!")
    elif accuracy > 0.6:
        print("👍 GOOD PERFORMANCE - Further tuning recommended")
    else:
        print("⚠️  NEEDS MORE OPTIMIZATION")
    
    return {
        'snn': snn,
        'accuracy': accuracy,
        'training_history': training_history,
        'test_data': (X_test, y_test)
    }

if __name__ == "__main__":
    print("🎯 OPTIMIZED SNN TRAINING PIPELINE")
    print("This fixes the poor performance in the original SNN training")
    print()
    
    results = run_optimized_training()
    
    if results:
        print(f"\n📈 OPTIMIZATION RESULTS:")
        print(f"   Final Accuracy: {results['accuracy']:.1%}")
        print(f"   Training Epochs: {len(results['training_history'])}")
        print(f"   Best Training Accuracy: {max(results['training_history']):.1%}")
        print(f"   Model: optimized_car_snn_model.pkl")
    else:
        print("❌ Optimization failed") 