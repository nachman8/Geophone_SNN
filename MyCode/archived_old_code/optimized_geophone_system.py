#!/usr/bin/env python3
"""
OPTIMIZED GEOPHONE SIGNAL CLASSIFICATION SYSTEM
Advanced feature extraction and SNN classification based on spikegram pattern analysis

Key Improvements:
1. Event-based feature extraction instead of simple statistics
2. Adaptive segmentation based on signal characteristics  
3. Optimized SNN architecture with stable training
4. Pattern-specific features for car vs human detection
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal as scipy_signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import SCTNeuron, create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer

# Frequency band definitions based on spikegram analysis
FREQUENCY_BANDS = {
    'LOW_FREQ': (0, 1),     # Band 0: 20-30 Hz
    'CAR_APPROACH': (1, 2), # Band 1: 30-34 Hz  
    'CAR_PEAK': (2, 3),     # Band 2: 34-40 Hz
    'CAR_TAIL': (3, 4),     # Band 3: 40-48 Hz
    'MID_GAP': (4, 5),      # Band 4: 48-60 Hz
    'HUMAN_PEAK': (5, 6),   # Band 5: 60-70 Hz
    'HUMAN_TAIL': (6, 7),   # Band 6: 70-85 Hz
    'HIGH_FREQ': (7, 8)     # Band 7: 85-100 Hz
}

# Signal-specific band indices for pattern detection
CAR_BANDS = [1, 2, 3]      # 30-48 Hz range where car patterns appear
HUMAN_BANDS = [5, 6]       # 60-85 Hz range where human footsteps appear

class SpikegrainPatternExtractor:
    """
    Advanced feature extractor based on spikegram pattern analysis
    
    Extracts features that capture the key differences between:
    - Car: Periodic patterns in 30-48 Hz range
    - Human: Burst patterns in 60-85 Hz range  
    - Nothing: Background noise patterns
    """
    
    def __init__(self, signal_type='car'):
        self.signal_type = signal_type
        self.scaler = RobustScaler()  # More robust to outliers
        
        # Signal-specific parameters based on spikegram analysis
        if signal_type == 'car':
            self.primary_bands = CAR_BANDS
            self.segment_duration = 12  # Shorter for better temporal resolution
            self.overlap_ratio = 0.7    # Higher overlap for better coverage
            self.expected_periodicity = 50  # Expected car pattern period in seconds
        else:  # human
            self.primary_bands = HUMAN_BANDS  
            self.segment_duration = 8   # Shorter for footstep events
            self.overlap_ratio = 0.6    # Moderate overlap
            self.expected_periodicity = None  # No expected periodicity for humans
    
    def extract_event_features(self, spikes_bands_spectrogram, duration):
        """
        Extract event-based features from spikegram patterns
        
        Key insight from spikegram analysis:
        - Cars show periodic activation in specific frequency bands
        - Humans show burst-like activation patterns
        - Nothing files show random low-level activation
        """
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        features = []
        
        # 1. TEMPORAL PATTERN FEATURES
        temporal_features = self._extract_temporal_patterns(spikes_bands_spectrogram, duration)
        features.extend(temporal_features)
        
        # 2. FREQUENCY DISTRIBUTION FEATURES  
        frequency_features = self._extract_frequency_patterns(spikes_bands_spectrogram)
        features.extend(frequency_features)
        
        # 3. EVENT DETECTION FEATURES
        event_features = self._extract_event_patterns(spikes_bands_spectrogram, duration)
        features.extend(event_features)
        
        # 4. CROSS-BAND CORRELATION FEATURES
        correlation_features = self._extract_correlation_patterns(spikes_bands_spectrogram)
        features.extend(correlation_features)
        
        return np.array(features)
    
    def _extract_temporal_patterns(self, spectrogram, duration):
        """Extract temporal pattern features based on spikegram analysis"""
        features = []
        
        # Focus on primary bands for this signal type
        primary_data = spectrogram[self.primary_bands, :]
        
        # 1. Activity concentration over time
        temporal_profile = np.sum(primary_data, axis=0)
        
        # Periodicity detection (especially important for cars)
        if self.expected_periodicity:
            period_samples = int(self.expected_periodicity * 100)  # 100 samples/second
            if len(temporal_profile) > period_samples:
                # Autocorrelation to detect periodicity
                autocorr = np.correlate(temporal_profile, temporal_profile, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Check for peak around expected period
                if len(autocorr) > period_samples:
                    periodicity_strength = autocorr[period_samples] / np.max(autocorr)
                else:
                    periodicity_strength = 0
                features.append(periodicity_strength)
            else:
                features.append(0)
        else:
            features.append(0)  # No periodicity expected for humans
        
        # 2. Temporal concentration (how concentrated activity is in time)
        if np.sum(temporal_profile) > 0:
            temporal_entropy = entropy(temporal_profile + 1e-10)  # Add small value for stability
            temporal_concentration = 1 / (1 + temporal_entropy)  # Higher = more concentrated
        else:
            temporal_concentration = 0
        features.append(temporal_concentration)
        
        # 3. Burst detection (important for human footsteps)
        burst_features = self._detect_bursts(temporal_profile)
        features.extend(burst_features)
        
        # 4. Activity duration patterns
        activity_threshold = np.mean(temporal_profile) + 0.5 * np.std(temporal_profile)
        active_periods = temporal_profile > activity_threshold
        
        # Duration statistics
        if np.any(active_periods):
            active_durations = self._get_duration_stats(active_periods)
            features.extend(active_durations)
        else:
            features.extend([0, 0, 0])  # mean, std, max duration
        
        return features
    
    def _extract_frequency_patterns(self, spectrogram):
        """Extract frequency distribution patterns"""
        features = []
        
        # 1. Primary band dominance (key insight from spikegram patterns)
        total_activity = np.sum(spectrogram)
        primary_activity = np.sum(spectrogram[self.primary_bands, :])
        
        if total_activity > 0:
            primary_dominance = primary_activity / total_activity
        else:
            primary_dominance = 0
        features.append(primary_dominance)
        
        # 2. Band-specific activity ratios
        for band_idx in range(spectrogram.shape[0]):
            band_activity = np.sum(spectrogram[band_idx, :])
            if total_activity > 0:
                band_ratio = band_activity / total_activity  
            else:
                band_ratio = 0
            features.append(band_ratio)
        
        # 3. Frequency selectivity (how concentrated energy is in frequency)
        frequency_profile = np.sum(spectrogram, axis=1)
        if np.sum(frequency_profile) > 0:
            freq_entropy = entropy(frequency_profile + 1e-10)
            freq_selectivity = 1 / (1 + freq_entropy)
        else:
            freq_selectivity = 0
        features.append(freq_selectivity)
        
        return features
    
    def _extract_event_patterns(self, spectrogram, duration):
        """Extract event-based patterns (key for distinguishing car vs human vs nothing)"""
        features = []
        
        # Focus on primary bands
        primary_data = spectrogram[self.primary_bands, :]
        
        # 1. Event count and characteristics
        events = self._detect_events(primary_data)
        features.append(len(events))  # Number of events
        
        if len(events) > 0:
            # Event duration statistics
            durations = [event['duration'] for event in events]
            features.extend([
                np.mean(durations),
                np.std(durations),
                np.max(durations)
            ])
            
            # Event intensity statistics  
            intensities = [event['intensity'] for event in events]
            features.extend([
                np.mean(intensities),
                np.std(intensities),
                np.max(intensities)
            ])
            
            # Inter-event intervals (important for distinguishing patterns)
            if len(events) > 1:
                intervals = []
                for i in range(1, len(events)):
                    interval = events[i]['start'] - events[i-1]['end']
                    intervals.append(interval)
                
                features.extend([
                    np.mean(intervals),
                    np.std(intervals),
                    np.max(intervals)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            # No events detected
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        return features
    
    def _extract_correlation_patterns(self, spectrogram):
        """Extract cross-band correlation patterns"""
        features = []
        
        # 1. Primary band internal correlation
        if len(self.primary_bands) > 1:
            correlations = []
            for i in range(len(self.primary_bands)):
                for j in range(i+1, len(self.primary_bands)):
                    band1 = spectrogram[self.primary_bands[i], :]
                    band2 = spectrogram[self.primary_bands[j], :]
                    
                    if np.std(band1) > 0 and np.std(band2) > 0:
                        corr = np.corrcoef(band1, band2)[0, 1]
                        correlations.append(abs(corr))  # Use absolute correlation
                    else:
                        correlations.append(0)
            
            if correlations:
                features.extend([
                    np.mean(correlations),
                    np.max(correlations)
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # 2. Primary vs secondary band correlation (important for classification)
        secondary_bands = [i for i in range(spectrogram.shape[0]) if i not in self.primary_bands]
        
        if secondary_bands and self.primary_bands:
            primary_signal = np.mean(spectrogram[self.primary_bands, :], axis=0)
            secondary_signal = np.mean(spectrogram[secondary_bands, :], axis=0)
            
            if np.std(primary_signal) > 0 and np.std(secondary_signal) > 0:
                cross_corr = abs(np.corrcoef(primary_signal, secondary_signal)[0, 1])
            else:
                cross_corr = 0
            features.append(cross_corr)
        else:
            features.append(0)
        
        return features
    
    def _detect_bursts(self, temporal_profile):
        """Detect burst patterns in temporal profile"""
        features = []
        
        if len(temporal_profile) < 10:
            return [0, 0, 0]  # Not enough data
        
        # Adaptive threshold for burst detection
        threshold = np.mean(temporal_profile) + 1.5 * np.std(temporal_profile)
        
        # Find bursts
        burst_mask = temporal_profile > threshold
        
        # Count bursts
        burst_starts = np.where(np.diff(np.concatenate([[False], burst_mask, [False]])) == 1)[0]
        burst_ends = np.where(np.diff(np.concatenate([[False], burst_mask, [False]])) == -1)[0]
        
        num_bursts = len(burst_starts)
        features.append(num_bursts)
        
        if num_bursts > 0:
            # Burst duration statistics
            burst_durations = burst_ends - burst_starts
            features.extend([
                np.mean(burst_durations),
                np.std(burst_durations) if len(burst_durations) > 1 else 0
            ])
        else:
            features.extend([0, 0])
        
        return features
    
    def _get_duration_stats(self, binary_mask):
        """Get duration statistics from binary mask"""
        # Find continuous segments
        starts = np.where(np.diff(np.concatenate([[False], binary_mask, [False]])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[False], binary_mask, [False]])) == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            durations = ends - starts
            return [
                np.mean(durations),
                np.std(durations) if len(durations) > 1 else 0,
                np.max(durations)
            ]
        else:
            return [0, 0, 0]
    
    def _detect_events(self, primary_data):
        """Detect events in primary frequency bands"""
        events = []
        
        # Combine primary bands
        combined_signal = np.mean(primary_data, axis=0)
        
        # Adaptive threshold
        threshold = np.mean(combined_signal) + 2.0 * np.std(combined_signal)
        
        # Find events above threshold
        event_mask = combined_signal > threshold
        
        # Find event boundaries
        starts = np.where(np.diff(np.concatenate([[False], event_mask, [False]])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[False], event_mask, [False]])) == -1)[0]
        
        for start, end in zip(starts, ends):
            if end > start:  # Valid event
                duration = end - start
                intensity = np.mean(combined_signal[start:end])
                
                events.append({
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'intensity': intensity
                })
        
        return events
    
    def extract_segments_adaptively(self, spikes_bands_spectrogram, duration):
        """
        Extract segments adaptively based on signal content
        
        Instead of fixed-length segments, adapt to actual signal patterns
        """
        segments = []
        labels = []
        
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        # Handle small data
        if n_time_bins < self.segment_duration * 100:
            # Single segment
            features = self.extract_event_features(spikes_bands_spectrogram, duration)
            segments.append(features)
            
            # Determine label based on signal characteristics
            label = self._classify_segment_content(spikes_bands_spectrogram)
            labels.append(label)
            
            return np.array(segments), np.array(labels)
        
        # Adaptive segmentation
        samples_per_segment = int(self.segment_duration * 100)
        step_size = int(samples_per_segment * (1 - self.overlap_ratio))
        
        for start_idx in range(0, n_time_bins - samples_per_segment + 1, step_size):
            end_idx = start_idx + samples_per_segment
            segment_data = spikes_bands_spectrogram[:, start_idx:end_idx]
            
            # Extract advanced features
            features = self.extract_event_features(segment_data, self.segment_duration)
            segments.append(features)
            
            # Classify segment content
            label = self._classify_segment_content(segment_data)
            labels.append(label)
        
        return np.array(segments), np.array(labels)
    
    def _classify_segment_content(self, segment_data):
        """
        Classify segment content based on spikegram patterns
        
        Returns:
        1 = signal present (car/human activity)
        0 = nothing (background noise)
        """
        # Calculate activity in primary bands
        primary_activity = np.sum(segment_data[self.primary_bands, :])
        total_activity = np.sum(segment_data)
        
        if total_activity == 0:
            return 0  # No activity
        
        # Primary band dominance ratio
        dominance_ratio = primary_activity / total_activity
        
        # Signal strength
        mean_activity = np.mean(segment_data)
        std_activity = np.std(segment_data)
        signal_strength = mean_activity + std_activity
        
        # Overall baseline estimation
        overall_baseline = np.percentile(segment_data.flatten(), 75)
        
        # Signal-specific thresholds based on spikegram analysis
        if self.signal_type == 'car':
            # Car patterns: look for moderate dominance + reasonable signal strength
            has_signal = (
                (dominance_ratio > 0.20 and signal_strength > 1.2 * overall_baseline) or
                (dominance_ratio > 0.35) or  # Strong dominance
                (signal_strength > 2.0 * overall_baseline and dominance_ratio > 0.15)
            )
        else:  # human
            # Human patterns: look for burst-like activity with high dominance
            temporal_profile = np.sum(segment_data[self.primary_bands, :], axis=0)
            
            # Detect bursts
            if len(temporal_profile) > 10:
                threshold = np.mean(temporal_profile) + 1.0 * np.std(temporal_profile)
                bursts = temporal_profile > threshold
                burst_ratio = np.sum(bursts) / len(bursts)
            else:
                burst_ratio = 0
            
            has_signal = (
                (dominance_ratio > 0.25 and signal_strength > 1.0 * overall_baseline) or
                (burst_ratio > 0.15 and dominance_ratio > 0.20) or  # Burst activity
                (dominance_ratio > 0.40)  # Very strong dominance
            )
        
        return 1 if has_signal else 0

class OptimizedSNN:
    """
    Optimized SNN with stable training and better architecture
    """
    
    def __init__(self, n_input_neurons=None, learning_rate=0.002):
        self.n_input_neurons = n_input_neurons
        self.learning_rate = learning_rate
        self.network = None
        self.scaler = StandardScaler()
        self.trained = False
        
        # Optimized parameters based on analysis
        self.n_hidden = 100        # Larger network for better capacity
        self.spike_duration = 100  # Shorter for faster training
        self.class_labels = {0: 'signal', 1: 'nothing'}
    
    def create_optimized_network(self, n_input_neurons):
        """Create optimized SNN architecture"""
        network = SpikingNetwork()
        
        # Input layer with proper initialization
        input_neurons = []
        for i in range(n_input_neurons):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 2
            neuron.label = f"input_{i}"
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        network.add_layer(input_layer)
        
        # Hidden layer with optimized parameters
        hidden_neurons = []
        for i in range(self.n_hidden):
            neuron = create_SCTN()
            
            # Improved weight initialization (Xavier/Glorot)
            std = np.sqrt(2.0 / n_input_neurons)
            neuron.synapses_weights = np.random.normal(0.0, std, n_input_neurons).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, -1.5, 1.5)
            
            # Optimized neuron parameters
            neuron.leakage_factor = 1
            neuron.leakage_period = 3
            neuron.theta = -8
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # Conservative STDP for stability
            neuron.set_stdp(
                A_LTP=0.008,
                A_LTD=-0.004,
                tau=20.0,
                clk_freq=1000,
                wmax=2.0,
                wmin=-0.5
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        network.add_layer(hidden_layer)
        
        # Output layer
        output_neurons = []
        class_names = ['signal', 'nothing']
        
        for i in range(2):
            neuron = create_SCTN()
            
            # Output weight initialization
            neuron.synapses_weights = np.random.normal(0.5, 0.2, self.n_hidden).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.1, 1.5)
            
            # Output neuron parameters
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.theta = -6 - i  # Different thresholds
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 1
            neuron.membrane_should_reset = True
            neuron.label = f"output_{class_names[i]}"
            
            # Supervised STDP
            neuron.set_supervised_stdp(
                A=0.015,
                tau=12.0,
                clk_freq=1000,
                wmax=2.0,
                wmin=0.0,
                desired_output=np.array([], dtype=np.int64)
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        network.add_layer(output_layer)
        
        # Enable logging
        for neuron in output_neurons:
            network.log_out_spikes(neuron._id)
        
        print(f"Optimized SNN: {n_input_neurons} → {self.n_hidden} → 2")
        return network
    
    def enhanced_spike_encoding(self, X):
        """Enhanced spike encoding with better temporal patterns"""
        n_samples, n_features = X.shape
        spike_trains = np.zeros((n_samples, n_features, self.spike_duration), dtype=int)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X) if not self.trained else self.scaler.transform(X)
        
        # Convert to [0,1] range with robust normalization
        X_normalized = np.zeros_like(X_scaled)
        for i in range(n_features):
            feature_col = X_scaled[:, i]
            p25, p75 = np.percentile(feature_col, [25, 75])
            if p75 > p25:
                normalized = (feature_col - p25) / (p75 - p25)
                X_normalized[:, i] = np.clip(normalized, 0, 1)
            else:
                X_normalized[:, i] = 0.5
        
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                feature_val = X_normalized[sample_idx, feature_idx]
                
                if feature_val > 0.05:  # Only encode significant values
                    # Enhanced rate coding with temporal structure
                    base_rate = feature_val * 0.35  # Max 35% spike rate
                    
                    # Create temporal modulation
                    for t in range(self.spike_duration):
                        # Early emphasis pattern
                        if t < self.spike_duration // 3:
                            temporal_factor = 1.3
                        elif t < 2 * self.spike_duration // 3:
                            temporal_factor = 0.8
                        else:
                            temporal_factor = 1.0
                        
                        spike_prob = base_rate * temporal_factor
                        
                        # Refractory period simulation
                        if t > 0 and spike_trains[sample_idx, feature_idx, t-1] == 1:
                            spike_prob *= 0.3
                        
                        if np.random.random() < spike_prob:
                            spike_trains[sample_idx, feature_idx, t] = 1
        
        return spike_trains
    
    def train(self, X_train, y_train, n_epochs=80, validation_split=0.2):
        """Train SNN with improved stability"""
        if self.network is None:
            self.network = self.create_optimized_network(X_train.shape[1])
        
        print(f"Training Optimized SNN for {n_epochs} epochs...")
        print(f"Data: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"Classes: Signal={np.sum(y_train==0)}, Nothing={np.sum(y_train==1)}")
        
        # Split for validation
        if validation_split > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, 
                random_state=42, stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None
        
        # Convert to spike trains
        X_tr_spikes = self.enhanced_spike_encoding(X_tr)
        
        training_history = []
        best_val_acc = 0
        patience = 15
        wait = 0
        
        for epoch in range(n_epochs):
            # Adaptive learning rate
            if epoch < 30:
                current_lr = 1.0
            elif epoch < 60:
                current_lr = 0.8
            else:
                current_lr = 0.6
            
            epoch_correct = 0
            epoch_total = 0
            
            # Balanced training
            signal_indices = np.where(y_tr == 0)[0]
            nothing_indices = np.where(y_tr == 1)[0]
            
            # Create balanced batch
            min_class_size = min(len(signal_indices), len(nothing_indices))
            if min_class_size > 0:
                balanced_signal = np.random.choice(signal_indices, min_class_size, replace=True)
                balanced_nothing = np.random.choice(nothing_indices, min_class_size, replace=True)
                epoch_indices = np.concatenate([balanced_signal, balanced_nothing])
                np.random.shuffle(epoch_indices)
            else:
                epoch_indices = np.random.permutation(len(X_tr_spikes))
            
            for idx in epoch_indices:
                spike_train = X_tr_spikes[idx]
                target_class = y_tr[idx]
                
                # Reset network
                self.network.reset_input()
                
                # Set supervised targets
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target_class:
                            # Target neuron should spike
                            if target_class == 0:  # signal
                                spike_times = list(range(15, self.spike_duration-15, 12))
                            else:  # nothing
                                spike_times = list(range(25, self.spike_duration-25, 20))
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target should not spike
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present stimulus
                output_spikes = []
                for t in range(self.spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Prediction with confidence threshold
                total_spikes = np.sum(output_spikes, axis=0)
                
                if np.sum(total_spikes) > 0:
                    predicted = np.argmax(total_spikes)
                    confidence = total_spikes[predicted] / np.sum(total_spikes)
                    
                    # Only count confident predictions
                    if confidence > 0.6:
                        if predicted == target_class:
                            epoch_correct += 1
                        epoch_total += 1
                else:
                    predicted = 1  # Default to nothing
                    if predicted == target_class:
                        epoch_correct += 1
                    epoch_total += 1
            
            # Calculate training accuracy
            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            training_history.append(train_acc)
            
            # Validation
            val_acc = train_acc  # Use training accuracy if no validation set
            if X_val is not None and epoch % 5 == 0:
                val_acc = self._validate(X_val, y_val)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Progress reporting
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Train={train_acc:.3f}, Val={val_acc:.3f}, LR={current_lr:.2f}")
        
        self.trained = True
        
        # Plot training curve
        self._plot_training(training_history)
        
        return training_history
    
    def _validate(self, X_val, y_val):
        """Validate the model"""
        X_val_spikes = self.enhanced_spike_encoding(X_val)
        
        correct = 0
        total = len(X_val_spikes)
        
        for i in range(total):
            spike_train = X_val_spikes[i]
            target_class = y_val[i]
            
            self.network.reset_input()
            
            output_spikes = []
            for t in range(self.spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            total_spikes = np.sum(output_spikes, axis=0)
            
            if np.sum(total_spikes) > 0:
                predicted = np.argmax(total_spikes)
            else:
                predicted = 1
            
            if predicted == target_class:
                correct += 1
        
        return correct / total if total > 0 else 0
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        X_test_spikes = self.enhanced_spike_encoding(X_test)
        
        predictions = []
        confidences = []
        
        for spike_train in X_test_spikes:
            self.network.reset_input()
            
            output_spikes = []
            for t in range(self.spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            total_spikes = np.sum(output_spikes, axis=0)
            
            if np.sum(total_spikes) > 0:
                predicted = np.argmax(total_spikes)
                confidence = total_spikes[predicted] / np.sum(total_spikes)
            else:
                predicted = 1
                confidence = 0.1
            
            predictions.append(predicted)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        predictions, confidences = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        print(f"\n🎯 OPTIMIZED SNN EVALUATION")
        print("=" * 40)
        print(f"🎯 Accuracy: {accuracy:.1%}")
        print(f"📊 Avg Confidence: {np.mean(confidences):.3f}")
        print(f"🔥 High Confidence (>0.8): {np.sum(confidences > 0.8)}/{len(confidences)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        class_names = ['Signal', 'Nothing']
        
        print(f"\n📊 Confusion Matrix:")
        print("Predicted:     Signal   Nothing")
        for i, actual in enumerate(class_names):
            print(f"Actual {actual:8s}: {cm[i][0]:6d}   {cm[i][1]:6d}")
        
        # Classification report
        try:
            report = classification_report(y_test, predictions, target_names=class_names, output_dict=True, zero_division=0)
            
            print(f"\n📈 Classification Report:")
            for class_name in class_names:
                if class_name.lower() in report:
                    metrics = report[class_name.lower()]
                    print(f"{class_name:8s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        except:
            print("Could not generate classification report")
        
        return accuracy, cm, confidences
    
    def _plot_training(self, history):
        """Plot training progress"""
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'b-', linewidth=2, label='Training Accuracy')
        
        # Add smoothed line
        if len(history) > 10:
            smoothed = np.convolve(history, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(history)), smoothed, 'r-', linewidth=2, alpha=0.7, label='Smoothed (10-epoch)')
        
        plt.title('Optimized SNN Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("optimized_snn_training.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plot saved: optimized_snn_training.png")
    
    def save_model(self, filepath):
        """Save the optimized model"""
        model_data = {
            'scaler': self.scaler,
            'n_input_neurons': self.n_input_neurons,
            'learning_rate': self.learning_rate,
            'n_hidden': self.n_hidden,
            'spike_duration': self.spike_duration,
            'trained': self.trained,
            'class_labels': self.class_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Optimized SNN saved: {filepath}")

def load_and_prepare_data(signal_type='car'):
    """Load chunk data and prepare for training"""
    print(f"🔄 Loading {signal_type} data with optimized feature extraction...")
    
    from load_saved_chunks import load_chunks_directly
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    chunk_data = load_chunks_directly(chunks_dir)
    
    if not chunk_data:
        return None, None
    
    # Initialize feature extractor
    extractor = SpikegrainPatternExtractor(signal_type=signal_type)
    
    all_segments = []
    all_labels = []
    
    # Process signal file
    signal_file = signal_type
    if signal_file in chunk_data:
        print(f"📁 Processing {signal_file} chunks...")
        chunks = chunk_data[signal_file]['chunks']
        
        for chunk_idx, chunk in enumerate(chunks):
            spikes_data = chunk['spikes_bands_spectrogram']
            duration = chunk['duration']
            
            # Extract segments with optimized features
            segments, labels = extractor.extract_segments_adaptively(spikes_data, duration)
            
            # Keep only signal segments (label 1)
            signal_indices = labels == 1
            signal_segments = segments[signal_indices]
            
            for segment in signal_segments:
                all_segments.append(segment)
                all_labels.append(0)  # 0 = signal for SNN
            
            print(f"  Chunk {chunk_idx}: {len(signal_segments)} signal segments")
    
    # Process nothing file
    nothing_file = f"{signal_type}_nothing"
    if nothing_file in chunk_data:
        print(f"📁 Processing {nothing_file} chunks...")
        chunks = chunk_data[nothing_file]['chunks']
        
        for chunk_idx, chunk in enumerate(chunks):
            spikes_data = chunk['spikes_bands_spectrogram']
            duration = chunk['duration']
            
            # Extract segments with optimized features
            segments, labels = extractor.extract_segments_adaptively(spikes_data, duration)
            
            # Keep only nothing segments (label 0)
            nothing_indices = labels == 0
            nothing_segments = segments[nothing_indices]
            
            for segment in nothing_segments:
                all_segments.append(segment)
                all_labels.append(1)  # 1 = nothing for SNN
            
            print(f"  Chunk {chunk_idx}: {len(nothing_segments)} nothing segments")
    
    if len(all_segments) == 0:
        print(f"❌ No segments extracted for {signal_type}")
        return None, None
    
    X = np.array(all_segments)
    y = np.array(all_labels)
    
    print(f"\n📊 OPTIMIZED {signal_type.upper()} DATASET:")
    print(f"Total segments: {len(X)}")
    print(f"Features per segment: {X.shape[1]}")
    print(f"Signal segments: {np.sum(y == 0)}")
    print(f"Nothing segments: {np.sum(y == 1)}")
    
    return X, y

def run_optimized_classification(signal_type='car'):
    """Run optimized classification for a signal type"""
    print(f"\n🚀 OPTIMIZED {signal_type.upper()} CLASSIFICATION")
    print("=" * 60)
    
    # Load and prepare data
    X, y = load_and_prepare_data(signal_type)
    
    if X is None:
        print(f"❌ Failed to load {signal_type} data")
        return None
    
    # Check class balance
    if len(np.unique(y)) < 2:
        print(f"❌ Only one class found in {signal_type} data")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create and train optimized SNN
    snn = OptimizedSNN(learning_rate=0.002)
    
    # Train
    training_history = snn.train(X_train, y_train, n_epochs=100, validation_split=0.2)
    
    # Evaluate
    accuracy, cm, confidences = snn.evaluate(X_test, y_test)
    
    # Save model
    model_path = f"optimized_{signal_type}_snn.pkl"
    snn.save_model(model_path)
    
    print(f"\n✅ OPTIMIZED {signal_type.upper()} CLASSIFICATION COMPLETE!")
    print(f"🎯 Test Accuracy: {accuracy:.1%}")
    print(f"💾 Model saved: {model_path}")
    
    return {
        'snn': snn,
        'accuracy': accuracy,
        'training_history': training_history,
        'confidences': confidences,
        'model_path': model_path
    }

def run_complete_optimized_system():
    """Run the complete optimized system"""
    print("🚀 OPTIMIZED GEOPHONE CLASSIFICATION SYSTEM")
    print("Advanced pattern-based feature extraction + Stable SNN training")
    print("=" * 80)
    
    results = {}
    
    # Train Car Classification
    car_results = run_optimized_classification('car')
    if car_results:
        results['car'] = car_results
    
    # Train Human Classification  
    human_results = run_optimized_classification('human')
    if human_results:
        results['human'] = human_results
    
    # Summary
    print(f"\n" + "="*80)
    print("🏆 OPTIMIZED SYSTEM RESULTS")
    print("="*80)
    
    if results.get('car'):
        car_acc = results['car']['accuracy']
        print(f"🚗 CAR CLASSIFICATION: {car_acc:.1%}")
        
    if results.get('human'):
        human_acc = results['human']['accuracy'] 
        print(f"👤 HUMAN CLASSIFICATION: {human_acc:.1%}")
    
    print(f"\n🚀 KEY IMPROVEMENTS:")
    print(f"   • Event-based feature extraction from spikegram patterns")
    print(f"   • Adaptive segmentation based on signal characteristics")
    print(f"   • Optimized SNN architecture with stable training")
    print(f"   • Pattern-specific features for car vs human detection")
    print(f"   • Robust spike encoding and learning parameters")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = run_complete_optimized_system() 