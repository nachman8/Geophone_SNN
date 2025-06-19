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
    
    def __init__(self, n_input_neurons=None, n_hidden=50, learning_rate=0.01):
        self.n_input_neurons = n_input_neurons
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.network = None
        self.input_frequencies = None
        self.trained = False
        
        # Classification labels
        self.class_labels = {0: 'car', 1: 'nothing'}
        
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
            neuron.leakage_factor = 3
            neuron.leakage_period = 10
            neuron.theta = -15
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # Add STDP learning
            neuron.set_stdp(
                A_LTP=0.01,
                A_LTD=-0.005,
                tau=20.0,
                clk_freq=1000,
                wmax=2.0,
                wmin=0.0
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer (2 neurons for car vs nothing classification)
        output_neurons = []
        class_names = ['car', 'nothing']
        for i in range(2):  # Only 2 classes for now
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
                A=0.02,
                tau=15.0,
                clk_freq=1000,
                wmax=2.0,
                wmin=0.0,
                desired_output=np.array([], dtype=np.int64)  # Will be set during training
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable spike logging for output neurons
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"SNN created: {n_input_neurons} input → {self.n_hidden} hidden → 3 output neurons")
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
    
    def prepare_training_data(self, car_file, car_nothing_file, duration_per_file=120):
        """
        Prepare training data from car and car_nothing files
        """
        print("Preparing training data for car vs nothing classification...")
        
        X_train_segments = []
        y_train_segments = []
        
        # Process car data
        print("Processing car data...")
        car_signal, car_time = load_and_prepare_data(car_file, 1000, duration_per_file)
        if car_signal is not None:
            # Get resonator grid
            clk_resonators = get_resonator_grid(car_file)
            
            # Process with resonators
            car_output = process_with_resonator_grid_parallel(
                car_signal, 1000, clk_resonators, car_time[-1], num_processes=10
            )
            
            # Create spikegrams
            car_spikes_spec, car_freqs = events_to_max_spectrogram(
                car_output, car_time[-1], clk_resonators, car_file
            )
            car_bands_spec = spikes_to_bands(car_spikes_spec, car_freqs)
            
            # Extract spike features
            car_features = self.extract_spike_features(car_bands_spec, car_time[-1])
            
            # Add to training data with label 0 (car)
            for segment in car_features:
                X_train_segments.append(segment)
                y_train_segments.append(0)  # car label
        
        # Process car_nothing data
        print("Processing car_nothing data...")
        nothing_signal, nothing_time = load_and_prepare_data(car_nothing_file, 1000, duration_per_file)
        if nothing_signal is not None:
            # Get resonator grid (use car grid for consistency)
            clk_resonators = get_resonator_grid(car_file)
            
            # Process with resonators
            nothing_output = process_with_resonator_grid_parallel(
                nothing_signal, 1000, clk_resonators, nothing_time[-1], num_processes=10
            )
            
            # Create spikegrams
            nothing_spikes_spec, nothing_freqs = events_to_max_spectrogram(
                nothing_output, nothing_time[-1], clk_resonators, car_file
            )
            nothing_bands_spec = spikes_to_bands(nothing_spikes_spec, nothing_freqs)
            
            # Extract spike features
            nothing_features = self.extract_spike_features(nothing_bands_spec, nothing_time[-1])
            
            # Add to training data with label 1 (nothing)
            for segment in nothing_features:
                X_train_segments.append(segment)
                y_train_segments.append(1)  # nothing label
        
        X_train = np.array(X_train_segments)
        y_train = np.array(y_train_segments)
        
        print(f"Training data prepared: {len(X_train)} segments, {X_train.shape[1]} features per segment")
        print(f"Car segments: {np.sum(y_train == 0)}, Nothing segments: {np.sum(y_train == 1)}")
        
        return X_train, y_train
    
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
                            # This neuron should not spike
                            neuron.supervised_stdp.desired_output = np.array([], dtype=np.int64)
                
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
        class_names = ['Car', 'Nothing']  # Only 2 classes for now
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

def run_car_nothing_classification():
    """
    Run complete car vs nothing classification using SNN
    """
    print("🚗 STARTING CAR vs NOTHING SNN CLASSIFICATION")
    print("=" * 60)
    
    # File paths
    car_file = DATA_DIR / "car.csv"
    car_nothing_file = DATA_DIR / "car_nothing.csv"
    
    # Create SNN classifier
    snn = GeophoneSNN(n_hidden=30, learning_rate=0.01)
    
    # Prepare training data
    X, y = snn.prepare_training_data(car_file, car_nothing_file, duration_per_file=120)
    
    if len(X) == 0:
        print("❌ No training data available")
        return None
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train the SNN
    training_history = snn.train(X_train, y_train, n_epochs=100, spike_duration=150)
    
    # Evaluate on test set
    accuracy, report, cm = snn.evaluate(X_test, y_test)
    
    # Save the model
    model_path = "car_nothing_snn_model.pkl"
    snn.save_model(model_path)
    
    print(f"\n✅ Car vs Nothing SNN classification completed!")
    print(f"📊 Final Test Accuracy: {accuracy:.3f}")
    print(f"💾 Model saved to: {model_path}")
    
    return snn, accuracy, report

def create_enhanced_spike_encoding(X, spike_duration=200, encoding_type='rate'):
    """
    Enhanced spike encoding methods for better SNN performance
    Input X: shape (n_samples, n_features) where n_features = 8 bands × 7 features = 56
    """
    spike_trains = []
    
    # Normalize features to [0, 1] range for consistent encoding
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[1]):
        feature_col = X[:, i]
        if np.max(feature_col) > np.min(feature_col):
            X_normalized[:, i] = (feature_col - np.min(feature_col)) / (np.max(feature_col) - np.min(feature_col))
        else:
            X_normalized[:, i] = feature_col
    
    for sample in X_normalized:
        sample_spikes = []
        
        for i, feature_val in enumerate(sample):
            if encoding_type == 'rate':
                # Rate coding: higher values = higher spike rates
                spike_rate = np.clip(feature_val * 50, 0, 50)  # Scale to 0-50 Hz for stability
                
                # Generate regular spikes with some jitter
                if spike_rate > 0:
                    inter_spike_interval = 1000 / spike_rate  # ms
                    spike_times = []
                    t = 0
                    while t < spike_duration:
                        # Add jitter for more realistic spikes
                        jitter = np.random.normal(0, inter_spike_interval * 0.1)
                        t += inter_spike_interval + jitter
                        if t < spike_duration:
                            spike_times.append(t)
                else:
                    spike_times = []
                
                # Convert to binary spike train
                spike_train = np.zeros(spike_duration, dtype=int)
                for spike_time in spike_times:
                    idx = int(spike_time)
                    if 0 <= idx < spike_duration:
                        spike_train[idx] = 1
                        
            elif encoding_type == 'temporal':
                # Temporal coding: higher values = earlier spikes
                if feature_val > 0.1:  # Only encode significant values
                    # Map feature value to spike time (0-100ms)
                    spike_time = int((1.0 - np.clip(feature_val, 0, 1)) * 100)
                    spike_train = np.zeros(spike_duration, dtype=int)
                    if spike_time < spike_duration:
                        spike_train[spike_time] = 1
                else:
                    spike_train = np.zeros(spike_duration, dtype=int)
            
            sample_spikes.append(spike_train)
        
        spike_trains.append(np.array(sample_spikes))
    
    return spike_trains

def analyze_spectrograms_for_segments(spikes_bands_spectrogram, duration, signal_type='car'):
    """
    Analyze spectrograms to identify segments with significant activity
    Based on the frequency patterns we see in the provided spectrograms
    
    Car Data Patterns:
    - Clear periodic structure: Red vertical lines every ~50 seconds
    - Strong activity in 30-50 Hz range (CAR_APPROACH, CAR_PEAK, CAR_TAIL)
    - Temporal regularity: Very predictable, rhythmic patterns
    
    Human Data Patterns:  
    - Sporadic, event-based: Bursts of activity when footsteps occur
    - Strong activity in 60-85 Hz range (HUMAN_PEAK, HUMAN_TAIL)
    - Variable timing: Irregular intervals between events
    """
    n_bands, n_time_bins = spikes_bands_spectrogram.shape
    
    # Adaptive binning based on data type as recommended
    if signal_type == 'car':
        segment_duration = 15  # 10-20 second windows to capture periodic patterns (~50s cycles)
        threshold_multiplier = 1.2
        # Car signals: strong in 30-50 Hz range (bands 1,2,3,4)
        important_bands = [1, 2, 3, 4]  # CAR_APPROACH, CAR_PEAK, CAR_TAIL
    else:
        segment_duration = 7   # 5-10 second windows to capture individual footstep events
        threshold_multiplier = 1.5
        # Human signals: strong in 60-85 Hz range (bands 5,6,7)
        important_bands = [5, 6, 7]  # HUMAN_PEAK, HUMAN_TAIL, HIGH_FREQ
    
    samples_per_segment = int(segment_duration * 100)  # 100 samples per second
    
    segments = []
    segment_labels = []
    segment_confidence = []
    
    for start_idx in range(0, n_time_bins - samples_per_segment, samples_per_segment // 2):
        end_idx = start_idx + samples_per_segment
        segment_data = spikes_bands_spectrogram[:, start_idx:end_idx]
        
        # Calculate activity in important frequency bands
        important_activity = np.sum(segment_data[important_bands])
        total_activity = np.sum(segment_data)
        
        # Calculate signal strength
        mean_activity = np.mean(segment_data)
        std_activity = np.std(segment_data)
        signal_strength = mean_activity + std_activity
        
        # Determine if this segment contains the target signal
        activity_ratio = important_activity / (total_activity + 1e-10)
        
        # Advanced pattern detection based on signal type
        if signal_type == 'car':
            # Car: look for periodic patterns and strong mid-range frequencies (30-50 Hz)
            # Check for temporal regularity (consistent activity levels)
            temporal_consistency = np.std(np.mean(segment_data[important_bands], axis=0)) / (np.mean(segment_data[important_bands]) + 1e-10)
            periodic_score = 1.0 / (temporal_consistency + 0.1)  # Lower std = more periodic
            
            has_signal = (activity_ratio > 0.25) and (signal_strength > threshold_multiplier * np.mean(spikes_bands_spectrogram)) and (periodic_score > 2.0)
        else:
            # Human: look for burst patterns in high frequencies (60-85 Hz)
            # Check for event-based activity (peaks followed by quiet periods)
            activity_pattern = np.mean(segment_data[important_bands], axis=0)
            if len(activity_pattern) > 10:
                # Detect bursts: periods of high activity
                threshold = np.mean(activity_pattern) + 1.5 * np.std(activity_pattern)
                bursts = activity_pattern > threshold
                n_bursts = np.sum(np.diff(np.concatenate([[False], bursts, [False]])) == 1)
                burst_score = n_bursts / len(activity_pattern) * 100  # bursts per second
            else:
                burst_score = 0
            
            has_signal = (activity_ratio > 0.35) and (signal_strength > threshold_multiplier * np.mean(spikes_bands_spectrogram)) and (burst_score > 0.5)
        
        # Create feature vector for this segment
        segment_features = []
        for band_idx in range(n_bands):
            band_data = segment_data[band_idx]
            # Extract comprehensive features for each frequency band
            features = [
                np.mean(band_data),                    # Average activity
                np.max(band_data),                     # Peak activity  
                np.std(band_data),                     # Variability
                np.sum(band_data > np.mean(band_data) + np.std(band_data)),  # Spike count above threshold
                np.sum(band_data > 0) / len(band_data),  # Activity ratio
                np.percentile(band_data, 90),         # 90th percentile
                np.sum(np.diff(band_data) > 0) / len(band_data),  # Rising edge ratio
            ]
            segment_features.extend(features)
        
        segments.append(segment_features)
        segment_labels.append(1 if has_signal else 0)  # 1 = signal present, 0 = nothing
        segment_confidence.append(activity_ratio)
    
    return np.array(segments), np.array(segment_labels), np.array(segment_confidence)

def run_enhanced_car_classification():
    """
    Enhanced car vs nothing classification with improved spectral analysis
    """
    print("🚗 ENHANCED CAR vs NOTHING SNN CLASSIFICATION")
    print("=" * 60)
    
    # File paths
    car_file = DATA_DIR / "car.csv"
    car_nothing_file = DATA_DIR / "car_nothing.csv"
    
    print("🔍 Analyzing spectrograms for optimal segmentation...")
    
    # Create SNN classifier
    snn = GeophoneSNN(n_hidden=40, learning_rate=0.015)
    
    # Process both files and extract features
    all_segments = []
    all_labels = []
    
    # Process car file
    print("Processing car.csv...")
    car_signal, car_time = load_and_prepare_data(car_file, 1000, 120)
    if car_signal is not None:
        clk_resonators = get_resonator_grid(car_file)
        car_output = process_with_resonator_grid_parallel(
            car_signal, 1000, clk_resonators, car_time[-1], num_processes=10
        )
        car_spikes_spec, car_freqs = events_to_max_spectrogram(
            car_output, car_time[-1], clk_resonators, car_file
        )
        car_bands_spec = spikes_to_bands(car_spikes_spec, car_freqs)
        
        # Analyze spectrograms to identify car segments
        car_segments, car_seg_labels, car_confidence = analyze_spectrograms_for_segments(
            car_bands_spec, car_time[-1], 'car'
        )
        
        # Only keep segments with car activity
        car_indices = car_seg_labels == 1
        all_segments.extend(car_segments[car_indices])
        all_labels.extend([0] * np.sum(car_indices))  # 0 = car
        
        print(f"Found {np.sum(car_indices)} car segments out of {len(car_seg_labels)} total")
    
    # Process car_nothing file
    print("Processing car_nothing.csv...")
    nothing_signal, nothing_time = load_and_prepare_data(car_nothing_file, 1000, 120)
    if nothing_signal is not None:
        nothing_output = process_with_resonator_grid_parallel(
            nothing_signal, 1000, clk_resonators, nothing_time[-1], num_processes=10
        )
        nothing_spikes_spec, nothing_freqs = events_to_max_spectrogram(
            nothing_output, nothing_time[-1], clk_resonators, car_file
        )
        nothing_bands_spec = spikes_to_bands(nothing_spikes_spec, nothing_freqs)
        
        # Analyze spectrograms to identify nothing segments  
        nothing_segments, nothing_seg_labels, nothing_confidence = analyze_spectrograms_for_segments(
            nothing_bands_spec, nothing_time[-1], 'car'
        )
        
        # Only keep segments with no significant activity
        nothing_indices = nothing_seg_labels == 0
        all_segments.extend(nothing_segments[nothing_indices])
        all_labels.extend([1] * np.sum(nothing_indices))  # 1 = nothing
        
        print(f"Found {np.sum(nothing_indices)} nothing segments out of {len(nothing_seg_labels)} total")
    
    # Convert to arrays
    X = np.array(all_segments)
    y = np.array(all_labels)
    
    if len(X) == 0:
        print("❌ No training data available")
        return None
    
    print(f"Dataset: {len(X)} segments, {X.shape[1]} features")
    print(f"Car segments: {np.sum(y == 0)}, Nothing segments: {np.sum(y == 1)}")
    
    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Train with enhanced spike encoding
    print("🧠 Training SNN with enhanced spike encoding...")
    training_history = snn.train(X_train, y_train, n_epochs=80, spike_duration=200)
    
    # Evaluate
    accuracy, report, cm = snn.evaluate(X_test, y_test)
    
    # Save model
    model_path = "enhanced_car_snn_model.pkl"
    snn.save_model(model_path)
    
    print(f"\n✅ Enhanced Car SNN Classification Complete!")
    print(f"📊 Test Accuracy: {accuracy:.3f}")
    print(f"💾 Model saved: {model_path}")
    
    # Create visualization of results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training curve
    plt.subplot(2, 2, 1)
    plt.plot(training_history)
    plt.title('Training Accuracy')
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
    
    # Plot 3: Feature importance (approximate)
    predictions, confidence = snn.predict(X_test)
    plt.subplot(2, 2, 3)
    plt.hist(confidence[y_test == 0], alpha=0.7, label='Car', bins=20)
    plt.hist(confidence[y_test == 1], alpha=0.7, label='Nothing', bins=20)
    plt.title('Prediction Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 4: Class distribution
    plt.subplot(2, 2, 4)
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(['Car', 'Nothing'], counts)
    plt.title('Dataset Distribution')
    plt.ylabel('Number of Segments')
    
    plt.tight_layout()
    save_plot("snn_classification_results")
    
    return snn, accuracy, report, training_history

if __name__ == "__main__":
    # Run enhanced car vs nothing classification
    results = run_enhanced_car_classification()
    
    if results:
        snn, accuracy, report, history = results
        print(f"\n🎯 Final Results:")
        print(f"   Architecture: {snn.n_input_neurons} → {snn.n_hidden} → 3 neurons")
        print(f"   Learning Rule: Supervised STDP")
        print(f"   Test Accuracy: {accuracy:.1%}")
        print(f"   Training Epochs: {len(history)}")
        print(f"   Final Training Accuracy: {history[-1]:.1%}")
    else:
        print("❌ Classification failed") 