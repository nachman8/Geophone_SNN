#!/usr/bin/env python3
"""
Advanced SCTN Geophone Classification System
Based on proper usage patterns from sctnN library examples and resonator analysis
"""

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import sctnN components
from sctnN.spiking_network import SpikingNetwork
from sctnN.layers import SCTNLayer
from sctnN.spiking_neuron import create_SCTN, BINARY, IDENTITY
from sctnN.resonator_functions import get_closest_resonator

class GeophoneResonatorGrid:
    """
    Resonator-based frequency analysis for geophone signals
    Based on the resonator examples and time-frequency analysis patterns
    """
    
    def __init__(self):
        # Frequency bands specifically tuned for geophone analysis
        self.frequency_bands = {
            'LOW_FREQ': (20, 30),      # Background/environmental
            'CAR_APPROACH': (30, 34),  # Vehicle approach
            'CAR_PEAK': (34, 40),      # Main vehicle signature  
            'CAR_TAIL': (40, 48),      # Vehicle departure
            'MID_GAP': (48, 60),       # Transition band
            'HUMAN_PEAK': (60, 70),    # Primary footstep frequency
            'HUMAN_TAIL': (70, 85),    # Secondary footstep harmonics
            'HIGH_FREQ': (85, 100)     # High frequency noise
        }
        
        # Resonator frequencies optimized for geophone detection
        self.resonator_frequencies = [
            # Low frequency environmental (20-30 Hz)
            20.1, 22.1, 25.0, 27.9,
            # Car signature range (30-48 Hz) - CRITICAL for car detection
            30.5, 31.3, 33.7, 34.0, 36.0, 37.8, 39.6, 40.5, 41.2, 43.0, 46.0, 48.0,
            # Transition band (48-60 Hz)
            48.0, 51.0, 54.0, 58.0,
            # Human signature range (60-85 Hz) - CRITICAL for human detection
            60.0, 63.0, 66.0, 69.0, 70.0, 72.0, 75.0, 78.0, 81.0, 84.0,
            # High frequency (85-100 Hz)
            86.0, 89.0, 91.0, 95.0, 98.0
        ]
        
        # Clock frequency for resonators (from examples)
        self.clk_freq = 153600  # Standard from resonator examples
        
        # Signal processing parameters
        self.stabilization_time = 0.1  # 100ms stabilization
        self.temporal_resolution = 10  # 10ms time bins
        
        print(f"🎛️ GeophoneResonatorGrid initialized")
        print(f"   📊 {len(self.frequency_bands)} frequency bands")
        print(f"   🔊 {len(self.resonator_frequencies)} resonators")
        print(f"   ⏱️ {self.clk_freq} Hz clock frequency")

    def process_signal_with_resonators(self, signal, fs=1000, duration=None):
        """
        Process signal through resonator grid following sctnN examples
        """
        if duration is None:
            duration = len(signal) / fs
            
        print(f"🔄 Processing {duration:.1f}s signal through resonator grid...")
        
        # Resample signal to match clock frequency
        target_samples = int(duration * self.clk_freq)
        if len(signal) != target_samples:
            from scipy.signal import resample
            signal = resample(signal, target_samples)
        
        # Normalize signal amplitude (from examples)
        signal_amplitude = 1000  # Standard amplitude from examples
        normalized_signal = (signal / np.max(np.abs(signal))) * signal_amplitude
        
        # Process through each resonator
        resonator_outputs = {}
        
        for freq in self.resonator_frequencies:
            try:
                # Get resonator function for this frequency
                resonator_func, actual_freq = get_closest_resonator(freq)
                resonator = resonator_func()
                
                # Reset resonator state
                self._reset_resonator_state(resonator)
                
                # Enable spike logging (from examples)
                resonator.log_out_spikes(-1)
                
                # Stabilize resonator (from examples)
                stabilization_samples = int(self.clk_freq * self.stabilization_time)
                resonator.input_full_data(np.zeros(stabilization_samples))
                
                # Reset logs after stabilization
                output_neuron = resonator.neurons[-1]
                output_neuron.forget_logs()
                
                # Process actual signal
                resonator.input_full_data(normalized_signal)
                
                # Get output spikes
                output_spikes = output_neuron.out_spikes()
                resonator_outputs[freq] = output_spikes
                
            except Exception as e:
                print(f"⚠️ Error processing {freq} Hz resonator: {e}")
                resonator_outputs[freq] = np.array([])
        
        print(f"✅ Processed {len(resonator_outputs)} resonators")
        return self._convert_to_spectrogram(resonator_outputs, duration)
    
    def _reset_resonator_state(self, resonator):
        """Reset resonator state to prevent contamination"""
        for neuron in resonator.neurons:
            neuron.forget_logs()
            neuron.membrane_potential = 0
            neuron.leakage_timer = 0
            neuron.index = 0
    
    def _convert_to_spectrogram(self, resonator_outputs, duration):
        """
        Convert resonator spike outputs to time-frequency spectrogram
        Following the pattern from resonator examples
        """
        # Time binning (10ms resolution as in examples)
        n_time_bins = int(duration * 1000 / self.temporal_resolution)
        spectrogram = np.zeros((len(self.resonator_frequencies), n_time_bins))
        
        for i, freq in enumerate(self.resonator_frequencies):
            if freq in resonator_outputs and len(resonator_outputs[freq]) > 0:
                # Convert spike events to binned counts
                events = resonator_outputs[freq]
                spike_bins = self._events_to_bins(events, duration, n_time_bins)
                spectrogram[i, :] = spike_bins
        
        return self._group_into_bands(spectrogram)
    
    def _events_to_bins(self, events, duration, n_bins):
        """Convert spike events to time bins"""
        if len(events) == 0:
            return np.zeros(n_bins)
        
        # Convert to time (seconds)
        spike_times = events / self.clk_freq
        
        # Create time bins
        bin_edges = np.linspace(0, duration, n_bins + 1)
        spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
        
        return spike_counts.astype(float)
    
    def _group_into_bands(self, spectrogram):
        """Group resonators into frequency bands"""
        bands_spectrogram = np.zeros((len(self.frequency_bands), spectrogram.shape[1]))
        
        for i, (band_name, (fmin, fmax)) in enumerate(self.frequency_bands.items()):
            # Find resonators in this band
            band_indices = []
            for j, freq in enumerate(self.resonator_frequencies):
                if fmin <= freq < fmax:
                    band_indices.append(j)
            
            if band_indices:
                # Use maximum response in band (stronger signal detection)
                bands_spectrogram[i, :] = np.max(spectrogram[band_indices, :], axis=0)
        
        return bands_spectrogram


class SctnPatternExtractor:
    """
    Advanced pattern extraction from resonator spectrograms
    Based on spikegram analysis insights and signal processing patterns
    """
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, bands_spectrogram, duration, signal_type='unknown'):
        """
        Extract comprehensive features from band spectrogram
        """
        features = []
        self.feature_names = []
        
        # 1. TEMPORAL PATTERN FEATURES (inspired by getting_started.ipynb)
        temporal_features = self._extract_temporal_patterns(bands_spectrogram, signal_type)
        features.extend(temporal_features)
        
        # 2. FREQUENCY BAND FEATURES 
        band_features = self._extract_band_features(bands_spectrogram)
        features.extend(band_features)
        
        # 3. SIGNAL-SPECIFIC PATTERN FEATURES
        pattern_features = self._extract_signal_patterns(bands_spectrogram, signal_type)
        features.extend(pattern_features)
        
        # 4. CROSS-BAND CORRELATION FEATURES
        correlation_features = self._extract_correlation_features(bands_spectrogram)
        features.extend(correlation_features)
        
        return np.array(features)
    
    def _extract_temporal_patterns(self, spectrogram, signal_type):
        """Extract temporal patterns following spikegram analysis insights"""
        features = []
        
        # Overall temporal activity
        total_activity = np.sum(spectrogram, axis=0)
        
        # Basic statistics
        features.extend([
            np.mean(total_activity),
            np.std(total_activity), 
            np.max(total_activity),
            np.min(total_activity)
        ])
        self.feature_names.extend(['temp_mean', 'temp_std', 'temp_max', 'temp_min'])
        
        # Periodicity detection (critical for car detection)
        periodicity_strength = self._detect_periodicity(total_activity)
        features.append(periodicity_strength)
        self.feature_names.append('temp_periodicity')
        
        # Burst detection (critical for human detection)  
        burst_count, burst_intensity = self._detect_bursts(total_activity)
        features.extend([burst_count, burst_intensity])
        self.feature_names.extend(['temp_burst_count', 'temp_burst_intensity'])
        
        # Activity concentration
        if np.sum(total_activity) > 0:
            activity_entropy = self._calculate_entropy(total_activity)
            concentration = 1 / (1 + activity_entropy)
        else:
            concentration = 0
        features.append(concentration)
        self.feature_names.append('temp_concentration')
        
        return features
    
    def _extract_band_features(self, spectrogram):
        """Extract features for each frequency band"""
        features = []
        band_names = ['LOW_FREQ', 'CAR_APPROACH', 'CAR_PEAK', 'CAR_TAIL', 
                     'MID_GAP', 'HUMAN_PEAK', 'HUMAN_TAIL', 'HIGH_FREQ']
        
        total_energy = np.sum(spectrogram)
        
        for i, band_name in enumerate(band_names):
            band_data = spectrogram[i, :]
            
            # Band energy features
            band_energy = np.sum(band_data)
            band_ratio = band_energy / (total_energy + 1e-10)
            band_peak = np.max(band_data)
            band_activity = np.mean(band_data)
            
            features.extend([band_energy, band_ratio, band_peak, band_activity])
            self.feature_names.extend([
                f'{band_name}_energy', f'{band_name}_ratio', 
                f'{band_name}_peak', f'{band_name}_activity'
            ])
        
        return features
    
    def _extract_signal_patterns(self, spectrogram, signal_type):
        """Extract signal-specific patterns based on domain knowledge"""
        features = []
        
        # Car-specific patterns (30-48 Hz dominance, periodicity)
        car_bands = [1, 2, 3, 4]  # CAR_APPROACH to CAR_TAIL
        car_activity = np.sum(spectrogram[car_bands, :])
        car_dominance = car_activity / (np.sum(spectrogram) + 1e-10)
        
        # Human-specific patterns (60-85 Hz bursts)
        human_bands = [5, 6, 7]  # HUMAN_PEAK to HUMAN_TAIL  
        human_activity = np.sum(spectrogram[human_bands, :])
        human_dominance = human_activity / (np.sum(spectrogram) + 1e-10)
        
        # Background noise level
        noise_band = [0, 7]  # LOW_FREQ and HIGH_FREQ
        noise_level = np.mean(spectrogram[noise_band, :])
        
        features.extend([car_dominance, human_dominance, noise_level])
        self.feature_names.extend(['car_dominance', 'human_dominance', 'noise_level'])
        
        # Signal-to-noise ratio
        signal_bands = car_bands + human_bands
        signal_energy = np.sum(spectrogram[signal_bands, :])
        noise_energy = np.sum(spectrogram[noise_band, :])
        snr = signal_energy / (noise_energy + 1e-10)
        
        features.append(snr)
        self.feature_names.append('snr')
        
        return features
    
    def _extract_correlation_features(self, spectrogram):
        """Extract cross-band correlation features"""
        features = []
        
        # Correlations between adjacent bands
        for i in range(spectrogram.shape[0] - 1):
            corr = np.corrcoef(spectrogram[i, :], spectrogram[i+1, :])[0, 1]
            if np.isnan(corr):
                corr = 0
            features.append(corr)
            self.feature_names.append(f'corr_band_{i}_{i+1}')
        
        return features
    
    def _detect_periodicity(self, signal):
        """Detect periodicity in temporal signal (car signature)"""
        if len(signal) < 10:
            return 0
        
        # Autocorrelation for periodicity detection
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        if len(autocorr) > 50:  # Look for periods around 50 bins (5 seconds)
            period_strength = autocorr[50] / (np.max(autocorr) + 1e-10)
        else:
            period_strength = 0
        
        return period_strength
    
    def _detect_bursts(self, signal):
        """Detect burst patterns (human footsteps)"""
        if len(signal) == 0:
            return 0, 0
        
        # Define burst threshold
        threshold = np.mean(signal) + 2 * np.std(signal)
        
        # Find bursts
        above_threshold = signal > threshold
        burst_starts = np.diff(np.concatenate([[False], above_threshold, [False]])) == 1
        burst_count = np.sum(burst_starts)
        
        # Burst intensity
        if burst_count > 0:
            burst_intensity = np.mean(signal[above_threshold])
        else:
            burst_intensity = 0
        
        return burst_count, burst_intensity
    
    def _calculate_entropy(self, signal):
        """Calculate entropy of signal distribution"""
        hist, _ = np.histogram(signal, bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log(hist))
        return entropy


class SctnGeophoneClassifier:
    """
    SCTN-based geophone classifier using proper STDP learning
    Based on getting_started.ipynb STDP examples
    """
    
    def __init__(self, n_features=None, learning_rate=0.001):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.network = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # STDP parameters from getting_started.ipynb
        self.clk_freq = 153600
        self.A_LTP = 0.00025    # Positive learning rate
        self.A_LTD = -0.00015   # Negative learning rate  
        self.tau = self.clk_freq * 2.5e-3 / 2  # Time constant
        self.w_max = 100        # Maximum weight
        self.w_min = 0          # Minimum weight
        
        print(f"🧠 SctnGeophoneClassifier initialized")
        print(f"   📊 Learning rate: {learning_rate}")
        print(f"   🔧 STDP: A_LTP={self.A_LTP}, A_LTD={self.A_LTD}")
    
    def _build_network(self, n_features):
        """Build SCTN network following examples patterns"""
        self.network = SpikingNetwork()
        
        # Input layer - encode features as spikes (from examples)
        input_neurons = []
        for i in range(n_features):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.synapses_weights = np.array([1.0], dtype=np.float64)
            input_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(input_neurons))
        
        # Hidden layer with STDP learning
        hidden_size = max(8, n_features // 4)  # Adaptive hidden size
        hidden_neurons = []
        for i in range(hidden_size):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(1, 10, n_features).astype(np.float64)
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 50
            neuron.leakage_factor = 2
            neuron.leakage_period = 2
            neuron.theta = -0.1
            
            # Enable STDP learning (from examples)
            neuron.set_stdp(
                A_LTP=self.A_LTP,
                A_LTD=self.A_LTD, 
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=self.w_max,
                wmin=self.w_min
            )
            
            hidden_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(hidden_neurons))
        
        # Output layer (2 neurons: nothing=0, detection=1)
        output_neurons = []
        for i in range(2):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.uniform(1, 5, hidden_size).astype(np.float64)
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 30
            neuron.leakage_factor = 3
            neuron.leakage_period = 3
            neuron.theta = -5
            neuron.label = f"output_{i}"
            
            # Enable STDP learning
            neuron.set_stdp(
                A_LTP=self.A_LTP * 0.5,  # Reduced for output layer
                A_LTD=self.A_LTD * 0.5,
                tau=self.tau,
                clk_freq=self.clk_freq,
                wmax=self.w_max,
                wmin=self.w_min
            )
            
            output_neurons.append(neuron)
        
        self.network.add_layer(SCTNLayer(output_neurons))
        
        print(f"🏗️ Built SCTN network: {n_features} → {hidden_size} → 2")
        return self.network
    
    def _encode_features_to_spikes(self, features, spike_duration=100):
        """
        Encode feature vector to spike patterns
        Following the pattern from getting_started.ipynb
        """
        n_samples, n_features = features.shape
        spike_patterns = []
        
        for sample in features:
            # Normalize features to spike rates (0-1 range)
            normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-10)
            
            # Convert to spike probabilities
            spike_probs = np.clip(normalized, 0, 1)
            
            # Generate spike pattern
            sample_spikes = []
            for t in range(spike_duration):
                spikes = (np.random.random(n_features) < spike_probs).astype(int)
                sample_spikes.append(spikes)
            
            spike_patterns.append(np.array(sample_spikes))
        
        return spike_patterns
    
    def train(self, X, y, n_epochs=30, verbose=True):
        """
        Train SCTN network with STDP learning
        """
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Build network
        if self.network is None:
            self._build_network(X.shape[1])
        
        # Convert labels (detection=1, nothing=0) 
        y_binary = (y > 0).astype(int)
        
        # Encode features to spikes
        if verbose:
            print("🔄 Encoding features to spike patterns...")
        spike_patterns = self._encode_features_to_spikes(X)
        
        # Training loop
        training_accuracies = []
        
        for epoch in range(n_epochs):
            # Reset network state
            self.network.reset_input()
            
            correct = 0
            total = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X))
            
            for idx in indices:
                spikes = spike_patterns[idx]
                target = y_binary[idx]
                
                # Process through network
                outputs = []
                for spike_vector in spikes:
                    output = self.network.input(spike_vector)
                    outputs.append(output)
                
                # Get prediction from output spikes
                output_spikes = np.sum(outputs, axis=0)
                prediction = np.argmax(output_spikes)
                
                if prediction == target:
                    correct += 1
                total += 1
            
            accuracy = correct / total
            training_accuracies.append(accuracy)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: {accuracy:.1%}")
        
        self.is_trained = True
        return training_accuracies
    
    def predict(self, X):
        """Predict using trained SCTN network"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Normalize features
        X = self.scaler.transform(X)
        
        # Encode to spikes
        spike_patterns = self._encode_features_to_spikes(X)
        
        predictions = []
        
        for spikes in spike_patterns:
            # Reset network state
            self.network.reset_input()
            
            # Process through network
            outputs = []
            for spike_vector in spikes:
                output = self.network.input(spike_vector)
                outputs.append(output)
            
            # Get prediction
            output_spikes = np.sum(outputs, axis=0)
            prediction = np.argmax(output_spikes)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        y_binary = (y > 0).astype(int)
        
        accuracy = accuracy_score(y_binary, predictions)
        report = classification_report(y_binary, predictions, 
                                     target_names=['Nothing', 'Detection'])
        cm = confusion_matrix(y_binary, predictions)
        
        return accuracy, report, cm


def load_saved_chunks(chunks_base_dir):
    """Load existing chunk files"""
    print(f"🔄 Loading saved chunks from {chunks_base_dir}")
    
    chunk_data = {}
    
    for signal_type in ['car', 'car_nothing', 'human', 'human_nothing']:
        chunk_dir = os.path.join(chunks_base_dir, signal_type)
        index_file = os.path.join(chunk_dir, 'chunk_index.pkl')
        
        if os.path.exists(index_file):
            with open(index_file, 'rb') as f:
                chunk_index = pickle.load(f)
            
            chunks = []
            for chunk_file in chunk_index['chunk_files']:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk = pickle.load(f)
                    chunks.append(chunk)
            
            chunk_data[signal_type] = {
                'index': chunk_index,
                'chunks': chunks
            }
            
            print(f"   ✅ Loaded {len(chunks)} chunks for {signal_type}")
    
    return chunk_data


def main():
    """Main execution function"""
    print("🚀 ADVANCED SCTN GEOPHONE CLASSIFICATION SYSTEM")
    print("=" * 60)
    print("Based on proper sctnN library usage patterns and resonator analysis")
    print()
    
    # Initialize components
    resonator_grid = GeophoneResonatorGrid()
    pattern_extractor = SctnPatternExtractor()
    
    # Load chunk data
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    chunk_data = load_saved_chunks(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return
    
    # Process all chunks and extract features
    print("\n🔬 FEATURE EXTRACTION FROM CHUNKS")
    print("-" * 40)
    
    all_features = []
    all_labels = []
    
    for signal_type, data in chunk_data.items():
        print(f"\n📊 Processing {signal_type} chunks...")
        
        is_signal_type = not signal_type.endswith('_nothing')
        label = 1 if is_signal_type else 0
        
        for chunk_idx, chunk in enumerate(data['chunks']):
            if 'spikes_bands_spectrogram' not in chunk:
                continue
            
            # Extract features from this chunk's spectrogram
            spikes_bands_spectrogram = chunk['spikes_bands_spectrogram']
            duration = chunk.get('duration', 120)
            
            # Use the actual signal type for pattern extraction
            base_signal_type = signal_type.replace('_nothing', '')
            features = pattern_extractor.extract_features(
                spikes_bands_spectrogram, 
                duration, 
                base_signal_type
            )
            
            all_features.append(features)
            all_labels.append(label)
            
            print(f"   Chunk {chunk_idx}: {len(features)} features extracted")
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n📊 DATASET SUMMARY")
    print(f"   Total samples: {len(X)}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Detection samples: {np.sum(y == 1)}")
    print(f"   Nothing samples: {np.sum(y == 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Train SCTN classifier
    print(f"\n🧠 SCTN NETWORK TRAINING")
    print("-" * 40)
    
    classifier = SctnGeophoneClassifier(n_features=X.shape[1])
    training_history = classifier.train(X_train, y_train, n_epochs=50, verbose=True)
    
    # Evaluate
    print(f"\n📊 EVALUATION RESULTS")
    print("-" * 40)
    
    accuracy, report, cm = classifier.evaluate(X_test, y_test)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"Training Progress: {training_history[0]:.1%} → {training_history[-1]:.1%}")
    print(f"Improvement: {training_history[-1] - training_history[0]:+.1%}")
    
    print(f"\nClassification Report:")
    print(report)
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    if accuracy >= 0.85:
        print("🎉 EXCELLENT! Target accuracy achieved!")
    elif accuracy >= 0.70:
        print("✅ GOOD! Significant improvement achieved!")
    else:
        print("⚠️  Needs further optimization")
    
    return {
        'classifier': classifier,
        'accuracy': accuracy,
        'training_history': training_history,
        'test_results': (X_test, y_test),
        'resonator_grid': resonator_grid,
        'pattern_extractor': pattern_extractor
    }


if __name__ == "__main__":
    results = main() 