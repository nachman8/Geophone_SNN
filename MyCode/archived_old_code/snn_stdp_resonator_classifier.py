#!/usr/bin/env python3
"""
SNN STDP Resonator-Based Classification System
Direct resonator chunk processing with 5-fold cross-validation
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler

# Import sctnN components for SNN STDP
from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import SCTNeuron, create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer
from sctnN.learning_rules.stdp import STDP
from sctnN.learning_rules.supervised_stdp import SupervisedSTDP

class ResonatorFeatureExtractor:
    """
    Advanced feature extraction from resonator spikegram data
    Designed to work with any car/human data input
    """
    
    def __init__(self):
        self.band_names = [
            'LOW_FREQ', 'CAR_APPROACH', 'CAR_PEAK', 'CAR_TAIL', 
            'MID_GAP', 'HUMAN_PEAK', 'HUMAN_TAIL', 'HIGH_FREQ'
        ]
        
        # Frequency ranges for each band (Hz)
        self.band_ranges = [
            (20, 30), (30, 34), (34, 40), (40, 48),
            (48, 60), (60, 70), (70, 80), (90, 100)
        ]
        
        # Signal-specific band groups for pattern analysis
        self.car_bands = [0, 1, 2, 3]    # LOW_FREQ to CAR_TAIL (20-48 Hz)
        self.human_bands = [4, 5, 6]     # MID_GAP to HUMAN_TAIL (48-80 Hz)
        self.noise_bands = [7]           # HIGH_FREQ (90-100 Hz)
        
        print("🔧 ResonatorFeatureExtractor initialized")
        print(f"   📊 {len(self.band_names)} frequency bands")
        print(f"   🚗 Car bands: {self.car_bands} ({self.band_ranges[0][0]}-{self.band_ranges[3][1]} Hz)")
        print(f"   👤 Human bands: {self.human_bands} ({self.band_ranges[4][0]}-{self.band_ranges[6][1]} Hz)")
    
    def extract_resonator_features(self, spikes_bands_spectrogram, duration, signal_type='unknown'):
        """
        Extract comprehensive features from resonator spikegram data
        
        WHY THIS APPROACH:
        1. Direct resonator output analysis - preserves temporal spike patterns
        2. Multi-scale feature extraction - captures both local and global patterns
        3. Signal-type aware processing - optimized for car vs human characteristics
        4. Robust to varying input sizes - adaptive segmentation and normalization
        """
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        print(f"   📊 Processing spikegram: {n_bands} bands × {n_time_bins} time bins ({duration:.1f}s)")
        
        # Preprocess for robustness across different data
        cleaned_data = self._robust_preprocessing(spikes_bands_spectrogram)
        
        # Extract multiple feature categories
        features = {}
        
        # 1. Band-wise Spike Statistics (8 bands × 5 features = 40 features)
        features.update(self._extract_band_spike_features(cleaned_data))
        
        # 2. Temporal Spike Patterns (12 features)
        features.update(self._extract_temporal_spike_patterns(cleaned_data, duration))
        
        # 3. Cross-Band Spike Relationships (8 features)
        features.update(self._extract_cross_band_features(cleaned_data))
        
        # 4. Signal-Type Specific Features (10 features)
        features.update(self._extract_signal_specific_features(cleaned_data, signal_type))
        
        # Convert to feature vector
        feature_vector = []
        feature_names = []
        
        for category, category_features in features.items():
            for feature_name, feature_value in category_features.items():
                feature_vector.append(feature_value)
                feature_names.append(f"{category}_{feature_name}")
        
        # Robust handling of NaN/inf values
        feature_vector = np.array(feature_vector, dtype=np.float64)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"   ✅ Extracted {len(feature_vector)} features")
        return feature_vector, feature_names
    
    def _robust_preprocessing(self, spectrogram):
        """
        Robust preprocessing to handle different data characteristics
        """
        cleaned = np.copy(spectrogram).astype(np.float64)
        
        # Remove extreme outliers using percentile-based clipping
        for band_idx in range(spectrogram.shape[0]):
            band_data = spectrogram[band_idx]
            
            # Use 95th percentile as upper bound to handle varying signal strengths
            upper_bound = np.percentile(band_data, 95)
            cleaned[band_idx] = np.clip(band_data, 0, upper_bound)
        
        return cleaned
    
    def _extract_band_spike_features(self, spectrogram):
        """
        Extract spike-based features for each frequency band
        RATIONALE: Each frequency band captures different signal characteristics
        """
        features = {}
        
        for i, band_name in enumerate(self.band_names):
            band_data = spectrogram[i]
            
            # Basic spike statistics
            total_spikes = np.sum(band_data)
            mean_rate = np.mean(band_data)
            max_rate = np.max(band_data)
            std_rate = np.std(band_data)
            
            # Spike density (percentage of time bins with spikes)
            spike_density = np.mean(band_data > 0)
            
            features[f"band_{band_name}_total"] = total_spikes
            features[f"band_{band_name}_mean"] = mean_rate
            features[f"band_{band_name}_max"] = max_rate
            features[f"band_{band_name}_std"] = std_rate
            features[f"band_{band_name}_density"] = spike_density
        
        return {"band_spikes": features}
    
    def _extract_temporal_spike_patterns(self, spectrogram, duration):
        """
        Extract temporal patterns from spike data
        RATIONALE: Car signals show periodic patterns, human signals show burst patterns
        """
        features = {}
        
        # Overall temporal activity
        total_activity = np.sum(spectrogram, axis=0)
        
        # Basic temporal statistics
        features["temp_mean"] = np.mean(total_activity)
        features["temp_std"] = np.std(total_activity)
        features["temp_max"] = np.max(total_activity)
        
        # Temporal distribution characteristics
        if len(total_activity) > 0:
            features["temp_skewness"] = self._calculate_skewness(total_activity)
            features["temp_kurtosis"] = self._calculate_kurtosis(total_activity)
        else:
            features["temp_skewness"] = 0
            features["temp_kurtosis"] = 0
        
        # Periodicity detection for car signals
        features["temp_periodicity"] = self._detect_periodicity(total_activity)
        
        # Burst detection for human signals
        features["temp_burst_count"] = self._detect_bursts(total_activity)
        features["temp_burst_intensity"] = self._calculate_burst_intensity(total_activity)
        
        # Activity distribution over time
        features["temp_activity_spread"] = self._calculate_activity_spread(total_activity)
        
        # Quiet periods (important for distinguishing from noise)
        features["temp_quiet_ratio"] = self._calculate_quiet_ratio(total_activity)
        
        # Onset characteristics
        features["temp_onset_strength"] = self._calculate_onset_strength(total_activity)
        features["temp_sustained_activity"] = self._calculate_sustained_activity(total_activity)
        
        return {"temporal": features}
    
    def _extract_cross_band_features(self, spectrogram):
        """
        Extract relationships between frequency bands
        RATIONALE: Signal patterns span multiple frequencies with specific relationships
        """
        features = {}
        
        # Band activity ratios
        band_totals = np.sum(spectrogram, axis=1)
        total_activity = np.sum(band_totals)
        
        if total_activity > 0:
            # Car vs Human frequency dominance
            car_activity = np.sum(band_totals[self.car_bands])
            human_activity = np.sum(band_totals[self.human_bands])
            noise_activity = np.sum(band_totals[self.noise_bands])
            
            features["car_dominance"] = car_activity / total_activity
            features["human_dominance"] = human_activity / total_activity
            features["noise_ratio"] = noise_activity / total_activity
            
            # Frequency center of mass
            band_indices = np.arange(len(band_totals))
            features["freq_centroid"] = np.sum(band_indices * band_totals) / total_activity
            
            # Frequency spread
            features["freq_spread"] = np.sqrt(np.sum(((band_indices - features["freq_centroid"]) ** 2) * band_totals) / total_activity)
        else:
            features["car_dominance"] = 0
            features["human_dominance"] = 0
            features["noise_ratio"] = 0
            features["freq_centroid"] = 0
            features["freq_spread"] = 0
        
        # Band correlation (synchronization between bands)
        features["band_correlation"] = self._calculate_band_correlation(spectrogram)
        
        # Low vs high frequency ratio
        low_freq_activity = band_totals[0] + band_totals[1]  # LOW_FREQ + CAR_APPROACH
        high_freq_activity = band_totals[6] + band_totals[7]  # HUMAN_TAIL + HIGH_FREQ
        features["low_vs_high_ratio"] = low_freq_activity / (high_freq_activity + 1e-10)
        
        # Energy concentration (how concentrated energy is across bands)
        features["energy_concentration"] = self._calculate_energy_concentration(band_totals)
        
        return {"cross_band": features}
    
    def _extract_signal_specific_features(self, spectrogram, signal_type):
        """
        Extract features specific to expected signal characteristics
        RATIONALE: Car and human signals have distinct patterns that can be explicitly modeled
        """
        features = {}
        
        # Car-specific pattern analysis
        car_bands_data = spectrogram[self.car_bands]
        car_temporal = np.sum(car_bands_data, axis=0)
        
        features["car_consistency"] = 1 / (1 + np.std(car_temporal) / (np.mean(car_temporal) + 1e-10))
        features["car_periodicity"] = self._detect_periodicity(car_temporal)
        features["car_low_freq_preference"] = np.sum(spectrogram[0]) / (np.sum(spectrogram) + 1e-10)
        
        # Human-specific pattern analysis
        human_bands_data = spectrogram[self.human_bands]
        human_temporal = np.sum(human_bands_data, axis=0)
        
        features["human_variability"] = np.std(human_temporal) / (np.mean(human_temporal) + 1e-10)
        features["human_burst_score"] = self._calculate_burst_intensity(human_temporal)
        features["human_high_freq_preference"] = np.sum(spectrogram[5:7]) / (np.sum(spectrogram) + 1e-10)
        
        # Pattern stability over time
        features["pattern_stability"] = self._calculate_pattern_stability(spectrogram)
        
        # Signal vs noise characteristics
        features["signal_to_noise"] = self._calculate_signal_to_noise_ratio(spectrogram)
        
        # Harmonic structure (for identifying engine-like vs footstep-like patterns)
        features["harmonic_strength"] = self._calculate_harmonic_strength(spectrogram)
        
        # Impulse vs continuous characteristics
        features["impulse_score"] = self._calculate_impulse_score(spectrogram)
        
        return {"signal_specific": features}
    
    # Helper methods for feature calculation
    def _calculate_skewness(self, data):
        """Calculate skewness of data distribution"""
        if len(data) <= 1 or np.std(data) == 0:
            return 0
        
        mean_data = np.mean(data)
        std_data = np.std(data)
        return np.mean(((data - mean_data) / std_data) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data distribution"""
        if len(data) <= 1 or np.std(data) == 0:
            return 0
        
        mean_data = np.mean(data)
        std_data = np.std(data)
        return np.mean(((data - mean_data) / std_data) ** 4) - 3
    
    def _detect_periodicity(self, signal_data):
        """Detect periodic patterns using autocorrelation"""
        if len(signal_data) < 20:
            return 0
        
        # Normalize signal
        signal_norm = signal_data - np.mean(signal_data)
        if np.std(signal_norm) == 0:
            return 0
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find periodicity strength
        if len(autocorr) > 10:
            # Look for peaks in autocorrelation (excluding lag 0)
            max_lag = min(100, len(autocorr)//4)
            if max_lag > 1:
                peak_autocorr = np.max(autocorr[1:max_lag])
                return peak_autocorr / (autocorr[0] + 1e-10)
        return 0
    
    def _detect_bursts(self, signal_data):
        """Detect burst patterns (sudden increases in activity)"""
        if len(signal_data) < 10:
            return 0
        
        # Define burst threshold
        threshold = np.mean(signal_data) + 2 * np.std(signal_data)
        burst_points = signal_data > threshold
        
        # Count distinct bursts
        burst_count = 0
        in_burst = False
        
        for active in burst_points:
            if active and not in_burst:
                burst_count += 1
                in_burst = True
            elif not active:
                in_burst = False
        
        return burst_count
    
    def _calculate_burst_intensity(self, signal_data):
        """Calculate average intensity of burst events"""
        if len(signal_data) < 10:
            return 0
        
        threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
        burst_points = signal_data > threshold
        
        if np.sum(burst_points) > 0:
            return np.mean(signal_data[burst_points]) / (np.mean(signal_data) + 1e-10)
        return 0
    
    def _calculate_activity_spread(self, signal_data):
        """Calculate how spread out activity is over time"""
        if len(signal_data) == 0:
            return 0
        
        # Calculate temporal center of mass
        time_indices = np.arange(len(signal_data))
        total_activity = np.sum(signal_data)
        
        if total_activity == 0:
            return 0
        
        center_of_mass = np.sum(time_indices * signal_data) / total_activity
        spread = np.sqrt(np.sum(((time_indices - center_of_mass) ** 2) * signal_data) / total_activity)
        
        return spread / len(signal_data)  # Normalize by signal length
    
    def _calculate_quiet_ratio(self, signal_data):
        """Calculate ratio of quiet periods"""
        if len(signal_data) == 0:
            return 1
        
        threshold = np.mean(signal_data) * 0.1  # 10% of mean activity
        quiet_points = signal_data < threshold
        return np.mean(quiet_points)
    
    def _calculate_onset_strength(self, signal_data):
        """Calculate strength of signal onset"""
        if len(signal_data) < 10:
            return 0
        
        # Look at first 10% of signal
        onset_length = max(1, len(signal_data) // 10)
        onset_activity = np.mean(signal_data[:onset_length])
        overall_activity = np.mean(signal_data)
        
        return onset_activity / (overall_activity + 1e-10)
    
    def _calculate_sustained_activity(self, signal_data):
        """Calculate how sustained the activity is"""
        if len(signal_data) < 10:
            return 0
        
        # Coefficient of variation (lower = more sustained)
        cv = np.std(signal_data) / (np.mean(signal_data) + 1e-10)
        return 1 / (1 + cv)  # Convert to sustainability score
    
    def _calculate_band_correlation(self, spectrogram):
        """Calculate average correlation between frequency bands"""
        if spectrogram.shape[0] < 2 or spectrogram.shape[1] < 2:
            return 0
        
        correlations = []
        for i in range(spectrogram.shape[0]):
            for j in range(i+1, spectrogram.shape[0]):
                if np.std(spectrogram[i]) > 0 and np.std(spectrogram[j]) > 0:
                    corr = np.corrcoef(spectrogram[i], spectrogram[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0
    
    def _calculate_energy_concentration(self, band_totals):
        """Calculate how concentrated energy is across frequency bands"""
        total_energy = np.sum(band_totals)
        if total_energy == 0:
            return 0
        
        # Calculate entropy-based concentration
        probabilities = band_totals / total_energy
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) <= 1:
            return 1
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(probabilities))
        
        return 1 - (entropy / max_entropy)  # Higher value = more concentrated
    
    def _calculate_pattern_stability(self, spectrogram):
        """Calculate how stable patterns are over time"""
        if spectrogram.shape[1] < 10:
            return 0
        
        # Calculate correlation between consecutive time windows
        window_size = min(50, spectrogram.shape[1] // 4)
        correlations = []
        
        for i in range(spectrogram.shape[1] - window_size):
            window1 = spectrogram[:, i:i+window_size].flatten()
            window2 = spectrogram[:, i+1:i+1+window_size].flatten()
            
            if np.std(window1) > 0 and np.std(window2) > 0:
                corr = np.corrcoef(window1, window2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0
    
    def _calculate_signal_to_noise_ratio(self, spectrogram):
        """Calculate signal to noise ratio"""
        signal_bands = spectrogram[:7]  # All except HIGH_FREQ
        noise_bands = spectrogram[7:]   # HIGH_FREQ
        
        signal_power = np.mean(signal_bands)
        noise_power = np.mean(noise_bands)
        
        return signal_power / (noise_power + 1e-10)
    
    def _calculate_harmonic_strength(self, spectrogram):
        """Calculate strength of harmonic structure"""
        # Look for harmonic relationships between frequency bands
        fundamental_bands = spectrogram[:4]  # Lower frequency bands
        harmonic_bands = spectrogram[4:]     # Higher frequency bands
        
        fundamental_power = np.mean(fundamental_bands)
        harmonic_power = np.mean(harmonic_bands)
        
        # Strong harmonics show balanced power across frequency ranges
        if fundamental_power + harmonic_power == 0:
            return 0
        
        return 2 * fundamental_power * harmonic_power / ((fundamental_power + harmonic_power) ** 2)
    
    def _calculate_impulse_score(self, spectrogram):
        """Calculate how impulse-like vs continuous the signal is"""
        total_activity = np.sum(spectrogram, axis=0)
        
        if len(total_activity) == 0:
            return 0
        
        # Calculate peak-to-average ratio
        peak_activity = np.max(total_activity)
        avg_activity = np.mean(total_activity)
        
        return peak_activity / (avg_activity + 1e-10)

# Feature extractor will be completed in next part...

class SNNSTDPClassifier:
    """
    Spiking Neural Network with STDP learning for resonator-based classification
    Implements 5-fold cross-validation and detailed performance analysis
    """
    
    def __init__(self, n_hidden=60, learning_rate=0.01, stdp_params=None):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.network = None
        self.trained = False
        self.scaler = RobustScaler()
        
        # STDP parameters optimized for geophone signals
        self.stdp_params = stdp_params or {
            'A_LTP': 0.02,      # Long-term potentiation strength
            'A_LTD': -0.01,     # Long-term depression strength  
            'tau': 20.0,        # Time constant for STDP window
            'wmax': 3.0,        # Maximum synaptic weight
            'wmin': 0.1         # Minimum synaptic weight
        }
        
        # Spike encoding parameters
        self.spike_encoding = {
            'duration': 200,     # Spike train duration (ms)
            'max_rate': 100,     # Maximum firing rate (Hz)
            'encoding_type': 'rate_temporal'  # Rate coding with temporal structure
        }
        
        print(f"🧠 SNNSTDPClassifier initialized")
        print(f"   🔧 Hidden neurons: {self.n_hidden}")
        print(f"   📚 Learning rate: {self.learning_rate}")
        print(f"   ⚡ STDP: LTP={self.stdp_params['A_LTP']}, LTD={self.stdp_params['A_LTD']}")
        print(f"   🎯 Spike encoding: {self.spike_encoding['duration']}ms duration")
    
    def create_snn_network(self, n_input_features):
        """
        Create SNN with STDP learning
        
        ARCHITECTURE RATIONALE:
        1. Input layer: Direct feature encoding to spikes
        2. Hidden layer: STDP learning for pattern extraction
        3. Output layer: Supervised STDP for classification
        
        WHY STDP:
        - Biologically plausible learning
        - Temporal pattern sensitivity
        - Robust to noise and variations
        - Self-organizing feature detection
        """
        print(f"🏗️  Creating SNN network: {n_input_features} → {self.n_hidden} → 2")
        
        self.network = SpikingNetwork()
        
        # INPUT LAYER
        # Convert feature values to spike trains
        input_neurons = []
        for i in range(n_input_features):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 5  # Sensitive to input spikes
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.label = f"input_{i}"
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # HIDDEN LAYER WITH STDP
        # Pattern detection and feature learning
        hidden_neurons = []
        for i in range(self.n_hidden):
            neuron = create_SCTN()
            
            # Initialize weights with small random values
            neuron.synapses_weights = np.random.uniform(
                self.stdp_params['wmin'], 
                self.stdp_params['wmax'] * 0.3, 
                n_input_features
            ).astype(np.float64)
            
            # Neuron parameters optimized for pattern detection
            neuron.leakage_factor = 2      # Moderate leakage
            neuron.leakage_period = 5      # Temporal integration window
            neuron.theta = -15             # Firing threshold
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # STDP LEARNING CONFIGURATION
            # WHY THESE PARAMETERS:
            # - A_LTP > |A_LTD|: Favor strengthening useful connections
            # - tau=20ms: Matches typical neural response times
            # - wmax/wmin: Prevent weight saturation/elimination
            neuron.set_stdp(
                A_LTP=self.stdp_params['A_LTP'],
                A_LTD=self.stdp_params['A_LTD'],
                tau=self.stdp_params['tau'],
                clk_freq=1000,
                wmax=self.stdp_params['wmax'],
                wmin=self.stdp_params['wmin']
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # OUTPUT LAYER WITH SUPERVISED STDP
        # Binary classification: signal vs nothing
        output_neurons = []
        class_names = ['signal', 'nothing']
        
        for i in range(2):
            neuron = create_SCTN()
            
            # Output layer weights
            neuron.synapses_weights = np.random.uniform(
                self.stdp_params['wmin'],
                self.stdp_params['wmax'] * 0.5,
                self.n_hidden
            ).astype(np.float64)
            
            # Output neuron parameters
            neuron.leakage_factor = 3      # Stronger integration
            neuron.leakage_period = 8      # Longer temporal window
            neuron.theta = -20             # Higher threshold for decision
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 10
            neuron.membrane_should_reset = True
            neuron.label = f"output_{class_names[i]}"
            
            # SUPERVISED STDP FOR CLASSIFICATION
            # Learns to associate patterns with target outputs
            neuron.set_supervised_stdp(
                A=self.stdp_params['A_LTP'] * 1.5,  # Stronger supervision
                tau=self.stdp_params['tau'],
                clk_freq=1000,
                wmax=self.stdp_params['wmax'],
                wmin=self.stdp_params['wmin'],
                desired_output=np.array([], dtype=np.int64)
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable spike logging for analysis
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"   ✅ SNN network created successfully")
        return self.network
    
    def encode_features_to_spikes(self, features):
        """
        Convert feature vectors to spike trains
        
        ENCODING STRATEGY:
        1. Rate coding: Feature magnitude → firing rate
        2. Temporal structure: Early spikes for strong features
        3. Noise injection: Robustness to variations
        
        WHY THIS ENCODING:
        - Preserves feature importance through firing rates
        - Temporal patterns enable STDP learning
        - Robust to feature scaling differences
        """
        n_samples, n_features = features.shape
        spike_duration = self.spike_encoding['duration']
        
        # Normalize features to [0, 1] range
        features_norm = self.scaler.fit_transform(features)
        features_norm = np.clip(features_norm, 0, None)  # Remove negative values
        features_norm = features_norm / (np.max(features_norm, axis=0) + 1e-10)
        
        # Create spike trains
        spike_trains = np.zeros((n_samples, n_features, spike_duration), dtype=np.int32)
        
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                feature_value = features_norm[sample_idx, feature_idx]
                
                if feature_value > 0:
                    # RATE CODING: Higher feature → higher firing rate
                    base_rate = feature_value * self.spike_encoding['max_rate'] / 1000.0  # Convert to probability per ms
                    
                    # TEMPORAL CODING: Strong features spike earlier
                    temporal_bias = 1.0 - (feature_value * 0.3)  # Reduce rate over time for strong features
                    
                    for t in range(spike_duration):
                        # Time-varying spike probability
                        time_factor = temporal_bias + (1 - temporal_bias) * (1 - t / spike_duration)
                        spike_prob = base_rate * time_factor
                        
                        # Add small amount of noise for robustness
                        noise = np.random.uniform(-0.02, 0.02)
                        final_prob = max(0, min(0.5, spike_prob + noise))
                        
                        # Generate spike
                        spike_trains[sample_idx, feature_idx, t] = 1 if np.random.random() < final_prob else 0
        
        return spike_trains
    
    def train_with_cross_validation(self, features, labels, n_folds=5, n_epochs=60):
        """
        Train SNN with 5-fold cross-validation
        
        CROSS-VALIDATION RATIONALE:
        1. Robust performance estimation
        2. Prevents overfitting
        3. Validates generalization across different data
        4. Identifies optimal hyperparameters
        """
        print(f"\n🎯 TRAINING SNN WITH {n_folds}-FOLD CROSS-VALIDATION")
        print("=" * 60)
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_accuracies': [],
            'fold_f1_scores': [],
            'fold_training_histories': [],
            'confusion_matrices': []
        }
        
        fold_idx = 1
        for train_idx, val_idx in skf.split(features, labels):
            print(f"\n�� FOLD {fold_idx}/{n_folds}")
            print("-" * 30)
            
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            print(f"   Training: {len(X_train)} samples")
            print(f"   Validation: {len(X_val)} samples")
            print(f"   Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
            
            # Create fresh network for this fold
            if self.network is None:
                self.create_snn_network(features.shape[1])
            else:
                # Reset network weights
                self._reset_network_weights()
            
            # Train on this fold
            training_history = self._train_single_fold(
                X_train, y_train, X_val, y_val, n_epochs, fold_idx
            )
            
            # Evaluate on validation set
            val_accuracy, val_f1, val_cm = self._evaluate_fold(X_val, y_val)
            
            # Store results
            cv_results['fold_accuracies'].append(val_accuracy)
            cv_results['fold_f1_scores'].append(val_f1)
            cv_results['fold_training_histories'].append(training_history)
            cv_results['confusion_matrices'].append(val_cm)
            
            print(f"   ✅ Fold {fold_idx} Results: Accuracy={val_accuracy:.3f}, F1={val_f1:.3f}")
            
            fold_idx += 1
        
        # Calculate overall cross-validation results
        mean_accuracy = np.mean(cv_results['fold_accuracies'])
        std_accuracy = np.std(cv_results['fold_accuracies'])
        mean_f1 = np.mean(cv_results['fold_f1_scores'])
        std_f1 = np.std(cv_results['fold_f1_scores'])
        
        print(f"\n🏆 CROSS-VALIDATION RESULTS:")
        print("=" * 40)
        print(f"📊 Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"📈 Mean F1 Score: {mean_f1:.3f} ± {std_f1:.3f}")
        print(f"📋 Individual Fold Accuracies: {[f'{acc:.3f}' for acc in cv_results['fold_accuracies']]}")
        
        # Final training on full dataset
        print(f"\n🔄 FINAL TRAINING ON COMPLETE DATASET")
        print("-" * 40)
        self._reset_network_weights()
        final_history = self._train_single_fold(features, labels, features, labels, n_epochs, 'final')
        
        self.trained = True
        
        # Overall results
        overall_results = {
            'cv_mean_accuracy': mean_accuracy,
            'cv_std_accuracy': std_accuracy,
            'cv_mean_f1': mean_f1,
            'cv_std_f1': std_f1,
            'cv_fold_results': cv_results,
            'final_training_history': final_history
        }
        
        return overall_results
    
    def _train_single_fold(self, X_train, y_train, X_val, y_val, n_epochs, fold_id):
        """
        Train SNN on a single fold with STDP learning
        """
        print(f"   🔥 Training fold {fold_id} for {n_epochs} epochs...")
        
        # Convert features to spike trains
        X_train_spikes = self.encode_features_to_spikes(X_train)
        spike_duration = self.spike_encoding['duration']
        
        training_history = {
            'train_accuracy': [],
            'train_loss': []
        }
        
        for epoch in range(n_epochs):
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train_spikes))
            
            for idx in indices:
                spike_train = X_train_spikes[idx]
                target_class = y_train[idx]
                
                # Reset network state
                self.network.reset_input()
                
                # Set supervised learning target
                self._set_learning_target(target_class, spike_duration)
                
                # Present spike train to network
                output_spikes = []
                for t in range(spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Calculate prediction
                total_output_spikes = np.sum(output_spikes, axis=0)
                predicted_class = np.argmax(total_output_spikes)
                
                # Update metrics
                correct = (predicted_class == target_class)
                epoch_correct += int(correct)
                epoch_total += 1
                epoch_loss += 0 if correct else 1
            
            # Calculate epoch metrics
            train_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            train_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
            
            training_history['train_accuracy'].append(train_accuracy)
            training_history['train_loss'].append(train_loss)
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"      Epoch {epoch:2d}: Accuracy={train_accuracy:.3f}, Loss={train_loss:.3f}")
        
        return training_history
    
    def _set_learning_target(self, target_class, spike_duration):
        """
        Set supervised STDP target for output neurons
        """
        output_neurons = self.network.layers_neurons[-1].neurons
        
        for neuron_idx, neuron in enumerate(output_neurons):
            if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                if neuron_idx == target_class:
                    # Target neuron should spike
                    target_spikes = list(range(20, spike_duration-20, 30))  # Regular spiking
                    neuron.supervised_stdp.desired_output = np.array(target_spikes, dtype=np.int64)
                else:
                    # Non-target neuron should not spike
                    neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
    
    def _reset_network_weights(self):
        """Reset network weights for new fold training"""
        if self.network is not None:
            for layer in self.network.layers_neurons:
                for neuron in layer.neurons:
                    if hasattr(neuron, 'synapses_weights'):
                        n_weights = len(neuron.synapses_weights)
                        neuron.synapses_weights = np.random.uniform(
                            self.stdp_params['wmin'],
                            self.stdp_params['wmax'] * 0.3,
                            n_weights
                        ).astype(np.float64)
    
    def _evaluate_fold(self, X_val, y_val):
        """Evaluate SNN performance on validation set"""
        # Convert to spike trains
        X_val_spikes = self.encode_features_to_spikes(X_val)
        
        predictions = []
        for spike_train in X_val_spikes:
            # Reset network
            self.network.reset_input()
            
            # Present spike train
            output_spikes = []
            for t in range(self.spike_encoding['duration']):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            # Get prediction
            total_output_spikes = np.sum(output_spikes, axis=0)
            predicted_class = np.argmax(total_output_spikes)
            predictions.append(predicted_class)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average='weighted')
        cm = confusion_matrix(y_val, predictions)
        
        return accuracy, f1, cm
    
    def predict(self, features):
        """Make predictions using trained SNN"""
        if not self.trained:
            raise ValueError("Network must be trained before prediction")
        
        # Convert to spike trains
        spike_trains = self.encode_features_to_spikes(features)
        
        predictions = []
        confidence_scores = []
        
        for spike_train in spike_trains:
            # Reset network
            self.network.reset_input()
            
            # Present spike train
            output_spikes = []
            for t in range(self.spike_encoding['duration']):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            # Calculate prediction and confidence
            total_output_spikes = np.sum(output_spikes, axis=0)
            predicted_class = np.argmax(total_output_spikes)
            
            # Confidence based on spike difference
            total_spikes = np.sum(total_output_spikes)
            if total_spikes > 0:
                confidence = total_output_spikes[predicted_class] / total_spikes
            else:
                confidence = 0.5
            
            predictions.append(predicted_class)
            confidence_scores.append(confidence)
        
        return np.array(predictions), np.array(confidence_scores)

# Continue with data loading and main execution...

class ResonatorDataLoader:
    """
    Load and process resonator chunk data for SNN training
    Handles car vs car_nothing and human vs human_nothing classification
    """
    
    def __init__(self, chunks_dir):
        self.chunks_dir = chunks_dir
        self.feature_extractor = ResonatorFeatureExtractor()
        
        print(f"📁 ResonatorDataLoader initialized")
        print(f"   📂 Chunks directory: {chunks_dir}")
    
    def load_signal_data(self, signal_type='car'):
        """
        Load signal and nothing data for binary classification
        
        DATA ORGANIZATION:
        - signal_type='car': loads car/ vs car_nothing/
        - signal_type='human': loads human/ vs human_nothing/
        
        LABELS:
        - 1 = signal (car or human)
        - 0 = nothing (background/noise)
        """
        print(f"\n📊 LOADING {signal_type.upper()} DATA")
        print("=" * 50)
        
        signal_dir = os.path.join(self.chunks_dir, signal_type)
        nothing_dir = os.path.join(self.chunks_dir, f"{signal_type}_nothing")
        
        all_features = []
        all_labels = []
        data_info = []
        
        # Load SIGNAL data (label = 1)
        if os.path.exists(signal_dir):
            print(f"🔄 Processing {signal_type} signal chunks...")
            signal_features, signal_count = self._load_chunk_directory(
                signal_dir, label=1, signal_type=signal_type
            )
            all_features.extend(signal_features)
            all_labels.extend([1] * len(signal_features))
            print(f"   ✅ Loaded {signal_count} chunks → {len(signal_features)} signal segments")
        
        # Load NOTHING data (label = 0)  
        if os.path.exists(nothing_dir):
            print(f"🔄 Processing {signal_type}_nothing chunks...")
            nothing_features, nothing_count = self._load_chunk_directory(
                nothing_dir, label=0, signal_type=f"{signal_type}_nothing"
            )
            all_features.extend(nothing_features)
            all_labels.extend([0] * len(nothing_features))
            print(f"   ✅ Loaded {nothing_count} chunks → {len(nothing_features)} nothing segments")
        
        # Convert to arrays
        X = np.array(all_features) if all_features else np.array([])
        y = np.array(all_labels) if all_labels else np.array([])
        
        print(f"\n📈 {signal_type.upper()} DATASET SUMMARY:")
        print("-" * 30)
        print(f"   Total segments: {len(X)}")
        if len(X) > 0:
            print(f"   Features per segment: {X.shape[1]}")
            print(f"   Signal segments (1): {np.sum(y == 1)}")
            print(f"   Nothing segments (0): {np.sum(y == 0)}")
            signal_pct = np.sum(y == 1) / len(y) * 100
            nothing_pct = np.sum(y == 0) / len(y) * 100
            print(f"   Class balance: {signal_pct:.1f}% signal, {nothing_pct:.1f}% nothing")
        
        return X, y
    
    def _load_chunk_directory(self, chunk_dir, label, signal_type):
        """Load all chunks from a directory"""
        features = []
        chunk_count = 0
        
        # Load chunk index
        index_file = os.path.join(chunk_dir, 'chunk_index.pkl')
        if not os.path.exists(index_file):
            print(f"   ⚠️  No chunk index found in {chunk_dir}")
            return features, chunk_count
        
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        # Process each chunk file
        for chunk_file in chunk_index['chunk_files']:
            if not os.path.exists(chunk_file):
                continue
            
            try:
                # Load chunk data
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                # Extract resonator features from spikegram
                spikegram = chunk_data['spikes_bands_spectrogram']
                duration = chunk_data['duration']
                
                # Extract segments with features
                segments = self._extract_segments_from_chunk(
                    spikegram, duration, signal_type.replace('_nothing', '')
                )
                
                features.extend(segments)
                chunk_count += 1
                
            except Exception as e:
                print(f"   ❌ Error loading chunk {chunk_file}: {e}")
                continue
        
        return features, chunk_count
    
    def _extract_segments_from_chunk(self, spikegram, duration, signal_type, segment_duration=10):
        """
        Extract overlapping segments from a chunk
        
        SEGMENTATION RATIONALE:
        - segment_duration=10s: Captures complete signal patterns
        - 50% overlap: Increases training data while preserving patterns
        - Adaptive to chunk size: Works with any duration input
        """
        n_bands, n_time_bins = spikegram.shape
        samples_per_segment = int(segment_duration * 100)  # 100 samples per second
        
        segments = []
        
        if n_time_bins < samples_per_segment:
            # Small chunk: use entire chunk as one segment
            features, _ = self.feature_extractor.extract_resonator_features(
                spikegram, duration, signal_type
            )
            segments.append(features)
        else:
            # Large chunk: extract overlapping segments
            step_size = samples_per_segment // 2  # 50% overlap
            
            for start_idx in range(0, n_time_bins - samples_per_segment + 1, step_size):
                end_idx = start_idx + samples_per_segment
                segment_spikegram = spikegram[:, start_idx:end_idx]
                
                # Extract features for this segment
                features, _ = self.feature_extractor.extract_resonator_features(
                    segment_spikegram, segment_duration, signal_type
                )
                segments.append(features)
        
        return segments

def run_comprehensive_snn_analysis():
    """
    Main function to run comprehensive SNN STDP analysis
    """
    print("🚀 COMPREHENSIVE SNN STDP RESONATOR CLASSIFICATION")
    print("=" * 80)
    print("🔬 Using direct resonator chunk data with 5-fold cross-validation")
    print("🧠 SNN with STDP learning for biologically-inspired classification")
    print("=" * 80)
    
    # Initialize components
    chunks_dir = "project/MyCode/chunked_output"
    data_loader = ResonatorDataLoader(chunks_dir)
    
    results = {}
    
    # CAR CLASSIFICATION: car vs car_nothing
    print(f"\n🚗 CAR vs CAR_NOTHING CLASSIFICATION")
    print("=" * 60)
    
    try:
        # Load car data
        car_X, car_y = data_loader.load_signal_data('car')
        
        if len(car_X) > 10:  # Minimum samples for cross-validation
            # Create SNN classifier
            car_snn = SNNSTDPClassifier(
                n_hidden=80,           # Sufficient capacity for pattern learning
                learning_rate=0.015,   # Balanced learning rate
                stdp_params={
                    'A_LTP': 0.025,    # Strong potentiation for car patterns
                    'A_LTD': -0.012,   # Moderate depression
                    'tau': 25.0,       # Slightly longer time window for car patterns
                    'wmax': 3.5,       # Higher max weight for stronger connections
                    'wmin': 0.1
                }
            )
            
            # Train with 5-fold cross-validation
            car_results = car_snn.train_with_cross_validation(
                car_X, car_y, 
                n_folds=5,
                n_epochs=80
            )
            
            results['car'] = {
                'snn': car_snn,
                'results': car_results,
                'data_shape': car_X.shape,
                'class_distribution': np.bincount(car_y)
            }
            
            print(f"\n✅ CAR CLASSIFICATION COMPLETE")
            print(f"   🎯 CV Accuracy: {car_results['cv_mean_accuracy']:.3f} ± {car_results['cv_std_accuracy']:.3f}")
            print(f"   📈 CV F1 Score: {car_results['cv_mean_f1']:.3f} ± {car_results['cv_std_f1']:.3f}")
            
        else:
            print("❌ Insufficient car data for training")
            results['car'] = None
            
    except Exception as e:
        print(f"❌ Car classification failed: {e}")
        import traceback
        traceback.print_exc()
        results['car'] = None
    
    # HUMAN CLASSIFICATION: human vs human_nothing
    print(f"\n\n👤 HUMAN vs HUMAN_NOTHING CLASSIFICATION")
    print("=" * 60)
    
    try:
        # Load human data
        human_X, human_y = data_loader.load_signal_data('human')
        
        if len(human_X) > 10:  # Minimum samples for cross-validation
            # Create SNN classifier
            human_snn = SNNSTDPClassifier(
                n_hidden=80,
                learning_rate=0.015,
                stdp_params={
                    'A_LTP': 0.020,    # Moderate potentiation for human patterns
                    'A_LTD': -0.015,   # Stronger depression for noise reduction
                    'tau': 20.0,       # Standard time window
                    'wmax': 3.0,       
                    'wmin': 0.1
                }
            )
            
            # Train with 5-fold cross-validation
            human_results = human_snn.train_with_cross_validation(
                human_X, human_y,
                n_folds=5, 
                n_epochs=80
            )
            
            results['human'] = {
                'snn': human_snn,
                'results': human_results,
                'data_shape': human_X.shape,
                'class_distribution': np.bincount(human_y)
            }
            
            print(f"\n✅ HUMAN CLASSIFICATION COMPLETE")
            print(f"   🎯 CV Accuracy: {human_results['cv_mean_accuracy']:.3f} ± {human_results['cv_std_accuracy']:.3f}")
            print(f"   📈 CV F1 Score: {human_results['cv_mean_f1']:.3f} ± {human_results['cv_std_f1']:.3f}")
            
        else:
            print("❌ Insufficient human data for training")
            results['human'] = None
            
    except Exception as e:
        print(f"❌ Human classification failed: {e}")
        import traceback
        traceback.print_exc()
        results['human'] = None
    
    # COMPREHENSIVE ANALYSIS AND REPORTING
    print(f"\n\n🎉 COMPREHENSIVE SNN STDP ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Performance Summary
    print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    for signal_type in ['car', 'human']:
        result = results.get(signal_type)
        if result and result['results']:
            res = result['results']
            data_shape = result['data_shape']
            class_dist = result['class_distribution']
            
            # Performance assessment
            accuracy = res['cv_mean_accuracy']
            if accuracy >= 0.90:
                status = "🟢 EXCELLENT"
            elif accuracy >= 0.80:
                status = "🟡 GOOD"
            elif accuracy >= 0.70:
                status = "🟠 FAIR"
            else:
                status = "🔴 NEEDS IMPROVEMENT"
            
            print(f"\n{status} {signal_type.upper()} CLASSIFICATION:")
            print(f"   📊 Dataset: {data_shape[0]} segments × {data_shape[1]} features")
            print(f"   📈 Classes: {class_dist[0]} nothing, {class_dist[1]} signal")
            print(f"   🎯 CV Accuracy: {res['cv_mean_accuracy']:.3f} ± {res['cv_std_accuracy']:.3f}")
            print(f"   📈 CV F1 Score: {res['cv_mean_f1']:.3f} ± {res['cv_std_f1']:.3f}")
            print(f"   🔬 Individual Folds: {[f'{acc:.3f}' for acc in res['cv_fold_results']['fold_accuracies']]}")
        else:
            print(f"\n❌ {signal_type.upper()} CLASSIFICATION: FAILED")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_snn_analysis()
    
    print(f"\n✨ Analysis complete! SNN STDP classification with 5-fold CV finished.")
    
    # Additional detailed explanation will be provided separately
    print(f"\n📚 For detailed explanations of feature extraction and SNN STDP implementation,")
    print(f"    see the comprehensive analysis report.")
