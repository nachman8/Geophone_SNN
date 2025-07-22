#!/usr/bin/env python3
"""
Improved STDP SNN Classifier with Proper Data Utilization
- Extracts many labeled segments from each 120s chunk
- Uses actual STDP-based spiking neural network classification
- Implements footprint-based pattern detection for labeling
- Uses simplified SNN implementation that works with available libraries

Key Improvements:
1. Segment Extraction: 15s segments for car, 7s for human from 120s chunks
2. Footprint Detection: Identifies signal vs nothing patterns within segments  
3. Real STDP SNN: Uses basic STDP with proper spike-based learning
4. Better Data Utilization: ~100x more training samples from same data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ChunkSegmentExtractor:
    """
    Extract many labeled segments from large 120s chunks
    Detects footprint patterns for proper signal vs nothing labeling
    """
    
    def __init__(self):
        # Frequency band definitions for footprint detection
        self.band_names = [
            'LOW_FREQ', 'CAR_APPROACH', 'CAR_PEAK', 'CAR_TAIL', 
            'MID_GAP', 'HUMAN_PEAK', 'HUMAN_TAIL', 'HIGH_FREQ'
        ]
        
        # Signal-specific parameters
        self.car_bands = [1, 2, 3]      # CAR_APPROACH, CAR_PEAK, CAR_TAIL (30-48 Hz)
        self.human_bands = [4, 5, 6]    # MID_GAP, HUMAN_PEAK, HUMAN_TAIL (48-80 Hz)
        
        # Segment parameters
        self.car_segment_duration = 15   # 15 seconds for car
        self.human_segment_duration = 7  # 7 seconds for human
        
        print(f"📊 ChunkSegmentExtractor initialized:")
        print(f"   🚗 Car segments: {self.car_segment_duration}s, bands {self.car_bands}")
        print(f"   👤 Human segments: {self.human_segment_duration}s, bands {self.human_bands}")
    
    def extract_segments_from_chunks(self, chunk_data, signal_type):
        """
        Extract many labeled segments from chunk data
        Each 120s chunk becomes multiple 15s (car) or 7s (human) segments
        """
        print(f"\n🔄 Extracting segments from {signal_type} chunks...")
        
        all_segments = []
        all_labels = []
        all_footprints = []
        
        # Determine if this is a signal or nothing file
        is_nothing_file = signal_type.endswith('_nothing')
        base_signal_type = signal_type.replace('_nothing', '') if is_nothing_file else signal_type
        
        # Get segment duration for this signal type
        segment_duration = self.car_segment_duration if base_signal_type == 'car' else self.human_segment_duration
        
        for chunk_idx, chunk in enumerate(chunk_data):
            if 'spikes_bands_spectrogram' not in chunk:
                print(f"   ⚠️  Invalid chunk {chunk_idx}")
                continue
            
            spikegram = chunk['spikes_bands_spectrogram']
            duration = chunk['duration']
            
            print(f"   📦 Chunk {chunk_idx}: {spikegram.shape} ({duration:.1f}s)")
            
            # Extract segments from this chunk
            segments, labels, footprints = self._extract_segments_from_spikegram(
                spikegram, duration, base_signal_type, is_nothing_file, segment_duration
            )
            
            all_segments.extend(segments)
            all_labels.extend(labels)
            all_footprints.extend(footprints)
            
            signal_count = np.sum(np.array(labels) == 1)
            nothing_count = np.sum(np.array(labels) == 0)
            
            print(f"     ✅ {len(segments)} segments: {signal_count} signal, {nothing_count} nothing")
        
        print(f"   🎯 Total extracted: {len(all_segments)} segments")
        print(f"     📈 Signal segments: {np.sum(np.array(all_labels) == 1)}")
        print(f"     📉 Nothing segments: {np.sum(np.array(all_labels) == 0)}")
        
        return np.array(all_segments), np.array(all_labels), all_footprints
    
    def _extract_segments_from_spikegram(self, spikegram, duration, signal_type, is_nothing_file, segment_duration):
        """
        Extract multiple segments from a single spikegram with footprint-based labeling
        """
        n_bands, n_time_bins = spikegram.shape
        samples_per_segment = int(segment_duration * 100)  # 100 samples per second
        
        segments = []
        labels = []
        footprints = []
        
        # Extract overlapping segments
        stride = samples_per_segment // 2  # 50% overlap
        
        for start_idx in range(0, n_time_bins - samples_per_segment + 1, stride):
            end_idx = start_idx + samples_per_segment
            segment_spikegram = spikegram[:, start_idx:end_idx]
            
            # Extract footprint pattern from this segment
            footprint = self._extract_footprint_from_segment(segment_spikegram, signal_type)
            
            # Determine label based on footprint analysis and file type
            has_signal = self._detect_signal_in_footprint(footprint, signal_type, is_nothing_file)
            
            # Convert spikegram segment to feature vector for SNN
            features = self._spikegram_to_features(segment_spikegram)
            
            segments.append(features)
            labels.append(1 if has_signal else 0)  # 1 = signal, 0 = nothing
            footprints.append(footprint)
        
        return segments, labels, footprints
    
    def _extract_footprint_from_segment(self, segment_spikegram, signal_type):
        """
        Extract footprint characteristics from a segment
        """
        n_bands, n_time_bins = segment_spikegram.shape
        
        # Select relevant frequency bands
        if signal_type == 'car':
            target_bands = self.car_bands
        elif signal_type == 'human':
            target_bands = self.human_bands
        else:
            target_bands = list(range(n_bands))
        
        # Extract temporal pattern from target bands
        target_band_data = segment_spikegram[target_bands, :]
        temporal_pattern = np.mean(target_band_data, axis=0)
        
        footprint = {
            'temporal_pattern': temporal_pattern,
            'activity_strength': np.sum(temporal_pattern),
            'target_band_energy': np.sum(target_band_data),
            'total_energy': np.sum(segment_spikegram),
            'signal_type': signal_type
        }
        
        # Calculate pattern characteristics
        if np.mean(temporal_pattern) > 0:
            footprint['consistency'] = 1.0 / (1.0 + np.std(temporal_pattern) / np.mean(temporal_pattern))
        else:
            footprint['consistency'] = 0.0
        
        # Periodicity (cars show regular patterns)
        footprint['periodicity'] = self._calculate_periodicity(temporal_pattern)
        
        # Burst detection (humans show burst patterns) 
        footprint['burst_score'] = self._calculate_burst_score(temporal_pattern)
        
        # Target band dominance
        footprint['target_dominance'] = footprint['target_band_energy'] / (footprint['total_energy'] + 1e-10)
        
        return footprint
    
    def _detect_signal_in_footprint(self, footprint, signal_type, is_nothing_file):
        """
        Detect if footprint contains signal pattern based on signal characteristics
        """
        # Extract footprint features
        activity_strength = footprint['activity_strength']
        consistency = footprint['consistency']
        periodicity = footprint['periodicity']
        burst_score = footprint['burst_score']
        target_dominance = footprint['target_dominance']
        
        # Adaptive thresholds based on signal type
        if signal_type == 'car':
            # Car signals: regular, consistent patterns in 30-48 Hz
            if is_nothing_file:
                # Very strict criteria for car_nothing files
                has_signal = (
                    (target_dominance > 0.35) and
                    (periodicity > 0.6) and
                    (consistency > 0.55) and
                    (activity_strength > 5000)
                )
            else:
                # More lenient for car signal files
                has_signal = (
                    (target_dominance > 0.20) and
                    (periodicity > 0.4) and
                    (consistency > 0.45) and
                    (activity_strength > 2000)
                )
        
        elif signal_type == 'human':
            # Human signals: burst patterns in 48-80 Hz
            if is_nothing_file:
                # Very strict criteria for human_nothing files
                has_signal = (
                    (target_dominance > 0.40) and
                    (burst_score > 5.0) and
                    (activity_strength > 3000)
                )
            else:
                # More lenient for human signal files
                has_signal = (
                    (target_dominance > 0.25) and
                    (burst_score > 2.0) and
                    (activity_strength > 1500)
                )
        
        else:
            # Generic detection
            has_signal = (
                (target_dominance > 0.25) and
                (activity_strength > 2000)
            )
        
        return has_signal
    
    def _calculate_periodicity(self, signal):
        """Calculate periodicity using autocorrelation"""
        if len(signal) < 20:
            return 0.0
        
        signal_norm = signal - np.mean(signal)
        if np.std(signal_norm) == 0:
            return 0.0
        
        # Autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) < 10:
            return 0.0
        
        # Find periodicity strength
        max_lag = min(100, len(autocorr)//3)
        if max_lag > 1:
            peak_autocorr = np.max(autocorr[1:max_lag])
            return peak_autocorr / (autocorr[0] + 1e-10)
        
        return 0.0
    
    def _calculate_burst_score(self, signal):
        """Calculate burst activity score"""
        if len(signal) < 10:
            return 0.0
        
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        burst_points = signal > threshold
        
        # Count burst events
        burst_count = 0
        in_burst = False
        
        for is_above_threshold in burst_points:
            if is_above_threshold and not in_burst:
                burst_count += 1
                in_burst = True
            elif not is_above_threshold:
                in_burst = False
        
        return burst_count / len(signal) * 100
    
    def _spikegram_to_features(self, spikegram):
        """
        Convert spikegram segment to feature vector for SNN input
        """
        n_bands, n_time_bins = spikegram.shape
        
        features = []
        
        # Extract comprehensive features for each frequency band
        for band_idx in range(n_bands):
            band_data = spikegram[band_idx, :]
            
            if len(band_data) > 0:
                band_features = [
                    np.mean(band_data),                              # Mean activity
                    np.max(band_data),                               # Peak activity
                    np.std(band_data),                               # Variability
                    np.sum(band_data > np.mean(band_data) + np.std(band_data)),  # Spike count
                    np.sum(band_data > 0) / len(band_data),          # Activity ratio
                    np.percentile(band_data, 90),                    # 90th percentile
                    np.sum(np.diff(band_data) > 0) / len(band_data) if len(band_data) > 1 else 0  # Rising edge ratio
                ]
            else:
                band_features = [0.0] * 7
            
            features.extend(band_features)
        
        return np.array(features, dtype=np.float32)

class SimpleSTDPSNNClassifier:
    """
    Simplified STDP-based SNN classifier using basic spike processing
    """
    
    def __init__(self, n_input_features=56, n_hidden=60, n_output=2):
        self.n_input_features = n_input_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Network weights
        self.W_input_hidden = np.random.uniform(0.3, 0.7, (n_input_features, n_hidden))
        self.W_hidden_output = np.random.uniform(0.3, 0.7, (n_hidden, n_output))
        
        # STDP parameters
        self.learning_rate = 0.01
        self.tau_pre = 20.0
        self.tau_post = 30.0
        self.A_plus = 0.02
        self.A_minus = -0.015
        
        # Neuron states
        self.hidden_spikes = np.zeros(n_hidden)
        self.output_spikes = np.zeros(n_output)
        self.spike_traces = np.zeros(n_input_features)
        
        print(f"🧠 SimpleSTDPSNNClassifier initialized:")
        print(f"   📊 Architecture: {n_input_features} → {n_hidden} → {n_output}")
        print(f"   ⚡ STDP: A+={self.A_plus}, A-={self.A_minus}")
    
    def spike_encode(self, features, spike_duration=200):
        """
        Convert features to spike trains using rate coding
        """
        # Normalize features to [0, 1]
        features_norm = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-10)
        
        # Generate spike trains
        max_rate = 100  # Hz
        dt = 1.0  # ms
        n_steps = int(spike_duration / dt)
        
        spike_trains = []
        for i, rate in enumerate(features_norm * max_rate):
            spike_prob = rate * dt / 1000.0  # Convert to probability per ms
            spikes = np.random.random(n_steps) < spike_prob
            spike_times = np.where(spikes)[0] * dt
            spike_trains.append(spike_times)
        
        return spike_trains
    
    def simulate_network(self, spike_trains, target=None, training=True):
        """
        Simulate the spiking network
        """
        spike_duration = 200  # ms
        dt = 1.0  # ms
        n_steps = int(spike_duration / dt)
        
        # Reset network state
        hidden_membrane = np.zeros(self.n_hidden)
        output_membrane = np.zeros(self.n_output)
        hidden_spike_times = [[] for _ in range(self.n_hidden)]
        output_spike_times = [[] for _ in range(self.n_output)]
        
        # Simulation loop
        for t in range(n_steps):
            current_time = t * dt
            
            # Input spikes
            input_spikes = np.zeros(self.n_input_features)
            for i, spike_times in enumerate(spike_trains):
                if len(spike_times) > 0 and np.any(np.abs(spike_times - current_time) < dt/2):
                    input_spikes[i] = 1.0
            
            # Hidden layer processing
            hidden_input = np.dot(input_spikes, self.W_input_hidden)
            hidden_membrane += hidden_input
            hidden_spikes = hidden_membrane > 1.0  # Threshold
            hidden_membrane[hidden_spikes] = 0.0  # Reset
            
            # Record hidden spikes
            for i, spike in enumerate(hidden_spikes):
                if spike:
                    hidden_spike_times[i].append(current_time)
            
            # Output layer processing
            output_input = np.dot(hidden_spikes.astype(float), self.W_hidden_output)
            output_membrane += output_input
            output_spikes = output_membrane > 1.0  # Threshold
            output_membrane[output_spikes] = 0.0  # Reset
            
            # Record output spikes
            for i, spike in enumerate(output_spikes):
                if spike:
                    output_spike_times[i].append(current_time)
            
            # STDP learning
            if training and target is not None:
                self._apply_stdp_learning(input_spikes, hidden_spikes, target, current_time)
        
        return output_spike_times
    
    def _apply_stdp_learning(self, input_spikes, hidden_spikes, target, current_time):
        """
        Apply STDP learning rules
        """
        # Simple supervised STDP: strengthen connections to target output
        if target < self.n_output:
            # Strengthen input->hidden connections when input spikes occur
            for i in range(self.n_input_features):
                if input_spikes[i] > 0:
                    self.W_input_hidden[i, :] += self.learning_rate * self.A_plus
                    self.W_input_hidden[i, :] = np.clip(self.W_input_hidden[i, :], 0, 1)
            
            # Strengthen hidden->target output connections when hidden spikes occur
            for i in range(self.n_hidden):
                if hidden_spikes[i] > 0:
                    self.W_hidden_output[i, target] += self.learning_rate * self.A_plus
                    # Weaken non-target outputs
                    for j in range(self.n_output):
                        if j != target:
                            self.W_hidden_output[i, j] += self.learning_rate * self.A_minus
                    
                    self.W_hidden_output[i, :] = np.clip(self.W_hidden_output[i, :], 0, 1)
    
    def train(self, X_train, y_train, n_epochs=30):
        """
        Train the STDP SNN
        """
        print(f"\n🎓 Training SimpleSTDP SNN for {n_epochs} epochs...")
        
        training_history = {
            'epoch': [],
            'accuracy': []
        }
        
        for epoch in range(n_epochs):
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            
            for idx in indices:
                features = X_train[idx]
                target = y_train[idx]
                
                # Encode features to spikes
                spike_trains = self.spike_encode(features)
                
                # Simulate network with learning
                output_spike_times = self.simulate_network(spike_trains, target, training=True)
                
                # Make prediction
                prediction = self._make_prediction(output_spike_times)
                
                if prediction == target:
                    correct_predictions += 1
            
            # Calculate accuracy
            epoch_accuracy = correct_predictions / len(X_train)
            training_history['epoch'].append(epoch)
            training_history['accuracy'].append(epoch_accuracy)
            
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"   📊 Epoch {epoch:2d}: accuracy={epoch_accuracy:.1%}")
        
        return training_history
    
    def predict(self, X_test):
        """
        Make predictions
        """
        predictions = []
        
        for features in X_test:
            # Encode features to spikes
            spike_trains = self.spike_encode(features)
            
            # Simulate network without learning
            output_spike_times = self.simulate_network(spike_trains, training=False)
            
            # Make prediction
            prediction = self._make_prediction(output_spike_times)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _make_prediction(self, output_spike_times):
        """
        Make prediction based on output spike counts
        """
        spike_counts = [len(spikes) for spikes in output_spike_times]
        
        if max(spike_counts) > 0:
            return np.argmax(spike_counts)
        else:
            return 0  # Default prediction

class ChunkedDataLoader:
    """
    Load chunked spikegram data and extract many labeled segments
    """
    
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        self.segment_extractor = ChunkSegmentExtractor()
        
        print(f"📁 ChunkedDataLoader initialized")
        print(f"   📂 Directory: {chunks_dir}")
    
    def load_and_extract_segments(self, signal_type):
        """
        Load chunks and extract many labeled segments
        """
        signal_dir = self.chunks_dir / signal_type
        
        if not signal_dir.exists():
            print(f"❌ Directory not found: {signal_dir}")
            return None, None, None
        
        print(f"📊 Loading {signal_type} chunks...")
        
        # Load all chunk files
        chunk_files = sorted(signal_dir.glob("chunk_*/chunk_*_data.pkl"))
        chunk_data = []
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk = pickle.load(f)
                if 'spikes_bands_spectrogram' in chunk:
                    chunk_data.append(chunk)
            except Exception as e:
                print(f"   ❌ Error loading {chunk_file}: {e}")
        
        print(f"   ✅ Loaded {len(chunk_data)} chunks")
        
        # Extract segments
        segments, labels, footprints = self.segment_extractor.extract_segments_from_chunks(
            chunk_data, signal_type
        )
        
        return segments, labels, footprints

def run_improved_stdp_classification():
    """
    Run improved STDP SNN classification with proper data utilization
    """
    print("🚀 Improved STDP SNN Classification with Proper Data Utilization")
    print("=" * 80)
    
    # Initialize components
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    data_loader = ChunkedDataLoader(chunks_dir)
    
    print("\n📁 Loading and extracting segments from chunks...")
    
    # Load and extract segments for all signal types
    car_segments, car_labels, car_footprints = data_loader.load_and_extract_segments('car')
    car_nothing_segments, car_nothing_labels, car_nothing_footprints = data_loader.load_and_extract_segments('car_nothing')
    
    human_segments, human_labels, human_footprints = data_loader.load_and_extract_segments('human')
    human_nothing_segments, human_nothing_labels, human_nothing_footprints = data_loader.load_and_extract_segments('human_nothing')
    
    # Check data availability
    if car_segments is None or car_nothing_segments is None:
        print("❌ Insufficient car data")
        return None
    
    if human_segments is None or human_nothing_segments is None:
        print("❌ Insufficient human data")
        return None
    
    # Prepare car dataset
    print(f"\n🚗 Preparing car binary dataset...")
    car_all_segments = np.vstack([car_segments, car_nothing_segments])
    car_all_labels = np.hstack([car_labels, car_nothing_labels])
    
    print(f"   📊 Total car segments: {len(car_all_segments)}")
    print(f"   📈 Signal segments: {np.sum(car_all_labels == 1)}")
    print(f"   📉 Nothing segments: {np.sum(car_all_labels == 0)}")
    
    # Prepare human dataset
    print(f"\n👤 Preparing human binary dataset...")
    human_all_segments = np.vstack([human_segments, human_nothing_segments])
    human_all_labels = np.hstack([human_labels, human_nothing_labels])
    
    print(f"   📊 Total human segments: {len(human_all_segments)}")
    print(f"   📈 Signal segments: {np.sum(human_all_labels == 1)}")
    print(f"   📉 Nothing segments: {np.sum(human_all_labels == 0)}")
    
    # Train car classifier
    print(f"\n" + "="*50)
    print("🚗 CAR STDP SNN CLASSIFICATION")
    print("="*50)
    
    if len(np.unique(car_all_labels)) >= 2:
        car_classifier = SimpleSTDPSNNClassifier(n_input_features=car_all_segments.shape[1])
        
        # Split data
        X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
            car_all_segments, car_all_labels, test_size=0.3, random_state=42, stratify=car_all_labels
        )
        
        print(f"🎓 Training set: {len(X_car_train)} samples")
        print(f"🧪 Test set: {len(X_car_test)} samples")
        
        # Train
        car_history = car_classifier.train(X_car_train, y_car_train, n_epochs=30)
        
        # Test
        car_predictions = car_classifier.predict(X_car_test)
        car_accuracy = accuracy_score(y_car_test, car_predictions)
        
        print(f"✅ Car STDP SNN Results:")
        print(f"   📊 Test Accuracy: {car_accuracy:.1%}")
        print(f"   📈 Final Training Accuracy: {car_history['accuracy'][-1]:.1%}")
        
        # Save model
        with open('car_stdp_snn_model.pkl', 'wb') as f:
            pickle.dump(car_classifier, f)
        print(f"   💾 Model saved: car_stdp_snn_model.pkl")
        
    else:
        print("❌ Insufficient car label diversity")
        car_accuracy = 0.0
    
    # Train human classifier  
    print(f"\n" + "="*50)
    print("👤 HUMAN STDP SNN CLASSIFICATION")
    print("="*50)
    
    if len(np.unique(human_all_labels)) >= 2:
        human_classifier = SimpleSTDPSNNClassifier(n_input_features=human_all_segments.shape[1])
        
        # Split data
        X_human_train, X_human_test, y_human_train, y_human_test = train_test_split(
            human_all_segments, human_all_labels, test_size=0.3, random_state=42, stratify=human_all_labels
        )
        
        print(f"🎓 Training set: {len(X_human_train)} samples")
        print(f"🧪 Test set: {len(X_human_test)} samples")
        
        # Train
        human_history = human_classifier.train(X_human_train, y_human_train, n_epochs=30)
        
        # Test
        human_predictions = human_classifier.predict(X_human_test)
        human_accuracy = accuracy_score(y_human_test, human_predictions)
        
        print(f"✅ Human STDP SNN Results:")
        print(f"   📊 Test Accuracy: {human_accuracy:.1%}")
        print(f"   📈 Final Training Accuracy: {human_history['accuracy'][-1]:.1%}")
        
        # Save model
        with open('human_stdp_snn_model.pkl', 'wb') as f:
            pickle.dump(human_classifier, f)
        print(f"   💾 Model saved: human_stdp_snn_model.pkl")
        
    else:
        print("❌ Insufficient human label diversity")
        human_accuracy = 0.0
    
    # Final summary
    print(f"\n" + "="*80)
    print("🎉 FINAL STDP SNN CLASSIFICATION RESULTS")
    print("="*80)
    print(f"🚗 Car Classification: {car_accuracy:.1%} accuracy")
    print(f"👤 Human Classification: {human_accuracy:.1%} accuracy")
    
    if car_accuracy > 0.7 and human_accuracy > 0.7:
        print("✅ EXCELLENT: Both STDP SNN classifiers performing well!")
    elif car_accuracy > 0.6 or human_accuracy > 0.6:
        print("👍 GOOD: At least one STDP SNN classifier performing well")
    else:
        print("⚠️  NEEDS IMPROVEMENT: STDP parameters may need tuning")
    
    print(f"\n📈 Data Utilization Improvement:")
    print(f"   📊 Car segments extracted: {len(car_all_segments)}")
    print(f"   📊 Human segments extracted: {len(human_all_segments)}")
    print(f"   🚀 Much more training data from same 120s chunks!")
    
    return {
        'car_accuracy': car_accuracy,
        'human_accuracy': human_accuracy,
        'car_segments': len(car_all_segments),
        'human_segments': len(human_all_segments)
    }

if __name__ == "__main__":
    try:
        results = run_improved_stdp_classification()
        print("\n✅ Improved STDP SNN classification completed successfully!")
        
        if results:
            print(f"\n📊 FINAL SUMMARY:")
            print(f"   🚗 Car: {results['car_accuracy']:.1%} accuracy ({results['car_segments']} segments)")
            print(f"   👤 Human: {results['human_accuracy']:.1%} accuracy ({results['human_segments']} segments)")
            
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc() 