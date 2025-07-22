#!/usr/bin/env python3
"""
Spiking Neural Network Classification for Geophone Signals
Using sctnN library for car/human/nothing detection
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
from sctnN.resonator_functions import get_closest_resonator

# Import our resonator processing functions
from resonator_work import (
    get_resonator_grid, load_and_prepare_data, process_with_resonator_grid_parallel,
    events_to_max_spectrogram, spikes_to_bands, save_plot, DATA_DIR
)

class GeophoneSNN:
    """
    Spiking Neural Network for geophone signal classification
    """
    
    def __init__(self, n_input_neurons=None, n_hidden=100, learning_rate=0.003):
        self.n_input_neurons = n_input_neurons
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.network = None
        self.input_frequencies = None
        self.trained = False
        
        # Classification labels for binary classification
        self.class_labels = {0: 'signal', 1: 'nothing'}
        
    def create_network(self, n_input_neurons):
        """
        Create the SNN architecture
        """
        self.n_input_neurons = n_input_neurons
        
        # Create the spiking network
        self.network = SpikingNetwork()
        
        # Input layer (resonator outputs)
        input_neurons = []
        for i in range(n_input_neurons):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 10
            neuron.label = f"input_{i}"
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Hidden layer
        hidden_neurons = []
        for i in range(self.n_hidden):
            neuron = create_SCTN()
            # Random initialization of synaptic weights
            neuron.synapses_weights = np.random.uniform(0.1, 1.0, n_input_neurons).astype(np.float64)
            neuron.leakage_factor = 1
            neuron.leakage_period = 3
            neuron.theta = -6
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # Add STDP learning
            neuron.set_stdp(
                A_LTP=0.02,
                A_LTD=-0.01,
                tau=15.0,
                clk_freq=1000,
                wmax=3.0,
                wmin=0.0
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer (2 neurons for binary classification)
        output_neurons = []
        class_names = ['signal', 'nothing']
        for i in range(2):  # 2 classes: signal vs nothing
            neuron = create_SCTN()
            # Random initialization of synaptic weights
            neuron.synapses_weights = np.random.uniform(0.1, 1.0, self.n_hidden).astype(np.float64)
            neuron.leakage_factor = 2
            neuron.leakage_period = 5
            neuron.theta = -10
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 8
            neuron.membrane_should_reset = True
            neuron.label = f"output_{class_names[i]}"
            
            # Add supervised STDP for classification
            # Initialize with empty desired output - will be set during training
            neuron.set_supervised_stdp(
                A=0.03,
                tau=10.0,
                clk_freq=1000,
                wmax=3.0,
                wmin=0.0,
                desired_output=np.array([], dtype=np.int64)  # Will be set during training
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable spike logging for output neurons
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"SNN created: {n_input_neurons} input → {self.n_hidden} hidden → 2 output neurons")
        return self.network
    
    def extract_spike_features(self, spikes_bands_spectrogram, duration, segment_duration=10):
        """
        Extract spike-based features from resonator outputs for SNN input
        """
        # Convert spectrogram to spike trains
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        # Segment the data into chunks
        samples_per_segment = int(segment_duration * 100)  # 100 samples per second
        n_segments = n_time_bins // samples_per_segment
        
        spike_features = []
        
        for segment_idx in range(n_segments):
            start_idx = segment_idx * samples_per_segment
            end_idx = start_idx + samples_per_segment
            
            segment_data = spikes_bands_spectrogram[:, start_idx:end_idx]
            
            # Convert to spike trains (threshold-based)
            spike_trains = []
            for band_idx in range(n_bands):
                band_data = segment_data[band_idx]
                
                # Adaptive thresholding
                threshold = np.mean(band_data) + 1.5 * np.std(band_data)
                spikes = (band_data > threshold).astype(int)
                
                # Convert to spike rate (spikes per second)
                spike_rate = np.sum(spikes) / segment_duration
                spike_trains.append(spike_rate)
            
            spike_features.append(spike_trains)
        
        return np.array(spike_features)
    
    def convert_to_spike_trains(self, X, spike_duration=100):
        """
        Convert feature vectors to spike trains for SNN input
        Uses enhanced spike encoding for better performance
        """
        return create_enhanced_spike_encoding(X, spike_duration, 'rate')
    
    def train(self, X_train, y_train, n_epochs=50, spike_duration=100):
        """
        Train the SNN using supervised STDP
        """
        if self.network is None:
            self.create_network(X_train.shape[1])
        
        print(f"Training SNN for {n_epochs} epochs...")
        
        # Convert to spike trains
        print("Converting training data to spike trains...")
        X_spike_trains = self.convert_to_spike_trains(X_train, spike_duration)
        
        training_accuracy = []
        
        for epoch in range(n_epochs):
            epoch_correct = 0
            epoch_total = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_spike_trains))
            
            for idx in indices:
                spike_train = X_spike_trains[idx]
                target_class = y_train[idx]
                
                # Reset network state
                self.network.reset_input()
                
                # Set desired output for supervised learning
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target_class:
                            # This neuron should spike during the stimulus
                            spike_times = list(range(10, spike_duration-10, 20))  # Spike every 20ms
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # This neuron should not spike - use -1 instead of empty array
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spike train to network
                output_spikes = []
                for t in range(spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Determine predicted class (neuron with most spikes)
                total_output_spikes = np.sum(output_spikes, axis=0)
                predicted_class = np.argmax(total_output_spikes)
                
                if predicted_class == target_class:
                    epoch_correct += 1
                epoch_total += 1
            
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            training_accuracy.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Training Accuracy = {accuracy:.3f}")
        
        self.trained = True
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(training_accuracy)
        plt.title('SNN Training Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        save_plot("snn_training_curve")
        
        print(f"Training completed. Final accuracy: {training_accuracy[-1]:.3f}")
        return training_accuracy
    
    def predict(self, X_test, spike_duration=100):
        """
        Make predictions using the trained SNN
        """
        if not self.trained:
            raise ValueError("Network must be trained before making predictions")
        
        # Convert to spike trains
        X_spike_trains = self.convert_to_spike_trains(X_test, spike_duration)
        
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
            
            # Determine predicted class
            total_output_spikes = np.sum(output_spikes, axis=0)
            predicted_class = np.argmax(total_output_spikes)
            confidence = total_output_spikes[predicted_class] / (np.sum(total_output_spikes) + 1e-10)
            
            predictions.append(predicted_class)
            confidence_scores.append(confidence)
        
        return np.array(predictions), np.array(confidence_scores)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained SNN
        """
        predictions, confidence = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        
        # Generate classification report
        class_names = ['Signal', 'Nothing']
        report = classification_report(
            y_test, predictions, 
            target_names=class_names[:len(np.unique(y_test))],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        print("\n" + "="*60)
        print("SNN CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Average Confidence: {np.mean(confidence):.3f}")
        
        print("\nConfusion Matrix:")
        print("Predicted:", class_names[:len(np.unique(y_test))])
        for i, actual_class in enumerate(class_names[:len(np.unique(y_test))]):
            print(f"Actual {actual_class:8s}: {cm[i]}")
        
        print("\nDetailed Classification Report:")
        for class_name in class_names[:len(np.unique(y_test))]:
            if class_name.lower() in report:
                metrics = report[class_name.lower()]
                print(f"{class_name:8s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return accuracy, report, cm
    
    def save_model(self, filepath):
        """Save the trained SNN model"""
        model_data = {
            'network_weights': self.get_network_weights(),
            'n_input_neurons': self.n_input_neurons,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'trained': self.trained,
            'class_labels': self.class_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"SNN model saved to {filepath}")
    
    def get_network_weights(self):
        """Extract weights from the network for saving"""
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

def prepare_chunked_training_data(chunk_indices_car, chunk_indices_nothing, signal_type='car'):
    """
    Prepare training data from chunked processing results
    """
    print("Preparing training data from chunked processing results...")
    
    X_train_segments = []
    y_train_segments = []
    
    # Import segment analysis function from resonator_work
    from resonator_work import analyze_spectrograms_for_segments
    
    # Process car chunks
    print("Processing car chunks...")
    car_segments_count = 0
    for chunk_index in chunk_indices_car:
        if 'nothing' not in chunk_index['file_path'].lower():  # Only actual car files
            # Load and process each chunk
            for chunk_idx, chunk_file in enumerate(chunk_index['chunk_files']):
                if not os.path.exists(chunk_file):
                    continue
                    
                try:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Extract segments from this chunk
                    chunk_segments, chunk_labels, _ = analyze_spectrograms_for_segments(
                        chunk_data['spikes_bands_spectrogram'],
                        chunk_data['duration'],
                        signal_type
                    )
                    
                    # Keep only segments with car activity (label 1 from analyze_spectrograms_for_segments)
                    car_indices = chunk_labels == 1
                    for segment in chunk_segments[car_indices]:
                        X_train_segments.append(segment)
                        y_train_segments.append(0)  # 0 = car for SNN
                        car_segments_count += 1
                        
                except Exception as e:
                    print(f"Error processing car chunk {chunk_file}: {e}")
                    continue
    
    # Process nothing chunks
    print("Processing nothing chunks...")
    nothing_segments_count = 0
    for chunk_index in chunk_indices_nothing:
        if 'nothing' in chunk_index['file_path'].lower():  # Only nothing files
            # Load and process each chunk
            for chunk_idx, chunk_file in enumerate(chunk_index['chunk_files']):
                if not os.path.exists(chunk_file):
                    continue
                    
                try:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Extract segments from this chunk
                    chunk_segments, chunk_labels, _ = analyze_spectrograms_for_segments(
                        chunk_data['spikes_bands_spectrogram'],
                        chunk_data['duration'],
                        signal_type
                    )
                    
                    # Keep only segments with no activity (label 0 from analyze_spectrograms_for_segments)
                    nothing_indices = chunk_labels == 0
                    for segment in chunk_segments[nothing_indices]:
                        X_train_segments.append(segment)
                        y_train_segments.append(1)  # 1 = nothing for SNN
                        nothing_segments_count += 1
                        
                except Exception as e:
                    print(f"Error processing nothing chunk {chunk_file}: {e}")
                    continue
    
    X_train = np.array(X_train_segments)
    y_train = np.array(y_train_segments)
    
    print(f"Chunked training data prepared:")
    print(f"  Total segments: {len(X_train)}")
    print(f"  Features per segment: {X_train.shape[1]}")
    print(f"  Car segments: {car_segments_count}")
    print(f"  Nothing segments: {nothing_segments_count}")
    
    return X_train, y_train

def run_chunked_car_classification(chunk_duration=120, num_processes=15):
    """
    Run car vs nothing classification using chunked processing
    """
    print("🚗 CHUNKED CAR vs NOTHING SNN CLASSIFICATION")
    print("=" * 60)
    
    # File paths
    car_file = DATA_DIR / "car.csv"
    car_nothing_file = DATA_DIR / "car_nothing.csv"
    
    print("🔄 Step 1: Processing files in chunks...")
    
    # Import chunked processing functions
    from resonator_work import process_file_in_chunks
    
    # Process car file in chunks
    print("Processing car.csv in chunks...")
    car_chunk_index = process_file_in_chunks(car_file, chunk_duration, num_processes)
    
    if car_chunk_index is None:
        print("❌ Failed to process car file")
        return None
    
    # Process car_nothing file in chunks
    print("Processing car_nothing.csv in chunks...")
    nothing_chunk_index = process_file_in_chunks(car_nothing_file, chunk_duration, num_processes)
    
    if nothing_chunk_index is None:
        print("❌ Failed to process car_nothing file")
        return None
    
    print("🧠 Step 2: Preparing training data from chunks...")
    
    # Prepare training data from chunks
    X, y = prepare_chunked_training_data([car_chunk_index], [nothing_chunk_index], 'car')
    
    if len(X) == 0:
        print("❌ No training data available")
        return None
    
    print("🎯 Step 3: Training SNN...")
    
    # Create SNN classifier
    snn = GeophoneSNN(n_hidden=40, learning_rate=0.015)
    
    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Train with enhanced spike encoding
    training_history = snn.train(X_train, y_train, n_epochs=80, spike_duration=200)
    
    # Evaluate
    accuracy, report, cm = snn.evaluate(X_test, y_test)
    
    # Save model
    model_path = "chunked_car_snn_model.pkl"
    snn.save_model(model_path)
    
    print(f"\n✅ Chunked Car SNN Classification Complete!")
    print(f"📊 Test Accuracy: {accuracy:.3f}")
    print(f"💾 Model saved: {model_path}")
    print(f"📁 Processed chunks: Car={len(car_chunk_index['chunk_files'])}, Nothing={len(nothing_chunk_index['chunk_files'])}")
    
    # Create visualization of results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training curve
    plt.subplot(2, 2, 1)
    plt.plot(training_history)
    plt.title('Training Accuracy (Chunked Data)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot 2: Confusion matrix
    plt.subplot(2, 2, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot 3: Confidence distribution
    predictions, confidence = snn.predict(X_test)
    plt.subplot(2, 2, 3)
    plt.hist(confidence[y_test == 0], alpha=0.7, label='Car', bins=20)
    plt.hist(confidence[y_test == 1], alpha=0.7, label='Nothing', bins=20)
    plt.title('Prediction Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 4: Chunk processing summary
    plt.subplot(2, 2, 4)
    chunk_counts = [len(car_chunk_index['chunk_files']), len(nothing_chunk_index['chunk_files'])]
    plt.bar(['Car Chunks', 'Nothing Chunks'], chunk_counts)
    plt.title('Chunks Processed')
    plt.ylabel('Number of Chunks')
    
    plt.tight_layout()
    save_plot("chunked_snn_classification_results")
    
    return {
        'snn': snn,
        'accuracy': accuracy,
        'report': report,
        'training_history': training_history,
        'car_chunks': len(car_chunk_index['chunk_files']),
        'nothing_chunks': len(nothing_chunk_index['chunk_files'])
    }

def create_enhanced_spike_encoding(X, spike_duration=200, encoding_type='rate'):
    """
    Enhanced spike encoding methods for better SNN performance
    Input X: shape (n_samples, n_features) where n_features = 8 bands × 7 features = 56
    """
    n_samples, n_features = X.shape
    spike_trains = np.zeros((n_samples, n_features, spike_duration), dtype=int)
    
    # Normalize features to [0, 1] range for consistent encoding
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[1]):
        feature_col = X[:, i]
        if np.max(feature_col) > np.min(feature_col):
            X_normalized[:, i] = (feature_col - np.min(feature_col)) / (np.max(feature_col) - np.min(feature_col))
        else:
            X_normalized[:, i] = feature_col
    
    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            feature_val = X_normalized[sample_idx, feature_idx]
            
            if encoding_type == 'rate':
                # Rate coding: higher values = higher spike rates
                spike_rate = np.clip(feature_val * 0.3, 0, 0.3)  # Max 30% chance of spike per time step
                
                # Generate spikes based on rate
                spikes = np.random.random(spike_duration) < spike_rate
                spike_trains[sample_idx, feature_idx] = spikes.astype(int)
                        
            elif encoding_type == 'temporal':
                # Temporal coding: higher values = earlier spikes
                if feature_val > 0.1:  # Only encode significant values
                    # Map feature value to spike time
                    spike_time = int((1.0 - np.clip(feature_val, 0, 1)) * (spike_duration - 1))
                    if 0 <= spike_time < spike_duration:
                        spike_trains[sample_idx, feature_idx, spike_time] = 1
    
    return spike_trains

# analyze_spectrograms_for_segments function moved to resonator_work.py to avoid circular imports

if __name__ == "__main__":
    print("🧠 CHUNKED SNN CLASSIFICATION FOR GEOPHONE SIGNALS")
    print("=" * 60)
    print("This approach uses chunked processing to handle large files")
    print("without memory issues while training the SNN classifier.")
    print()
    
    # Run chunked car vs nothing classification
    results = run_chunked_car_classification(
        chunk_duration=120,  # Process files in 120-second chunks
        num_processes=15     # Parallel processing
    )
    
    if results:
        print(f"\n🎯 CHUNKED SNN CLASSIFICATION RESULTS:")
        print(f"   🧠 Architecture: {results['snn'].n_input_neurons} → {results['snn'].n_hidden} → 2 neurons")
        print(f"   📚 Learning Rule: Supervised STDP")
        print(f"   📊 Test Accuracy: {results['accuracy']:.1%}")
        print(f"   🔄 Training Epochs: {len(results['training_history'])}")
        print(f"   📈 Final Training Accuracy: {results['training_history'][-1]:.1%}")
        print(f"   📁 Car chunks processed: {results['car_chunks']}")
        print(f"   📁 Nothing chunks processed: {results['nothing_chunks']}")
        print(f"   💾 Model saved: chunked_car_snn_model.pkl")
    else:
        print("❌ Chunked classification failed")
        
    print("\n" + "="*60)
    print("💡 MEMORY USAGE BENEFITS:")
    print("• Processes large files without memory overflow")
    print("• Maintains spectrogram quality for each chunk")
    print("• Extracts training segments from all chunks")
    print("• Compatible with existing SNN classification pipeline")
    print("=" * 60) 