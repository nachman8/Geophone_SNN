#!/usr/bin/env python3
"""
Advanced Footprint Detector
Implements sophisticated pattern detection based on:
1. EEG Mental Attention State Detection approaches
2. Supervised STDP Resonator techniques  
3. Range STDP analysis with phase detection
4. Semi-supervised burst pattern recognition

This module extracts the most discriminative signature patterns 
for car vs human footprint detection.
"""

import numpy as np
import pandas as pd
import pickle
from scipy.signal import find_peaks, correlate, periodogram
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFootprintDetector:
    """
    Advanced pattern detector for geophone footprint analysis
    Based on state-of-the-art approaches from neuroscience and signal processing
    """
    
    def __init__(self):
        self.car_signature_params = {
            'frequency_range': (30, 48),
            'periodicity_window': 500,
            'consistency_threshold': 0.7,
            'activity_bands': [1, 2, 3, 4]  # CAR bands
        }
        
        self.human_signature_params = {
            'frequency_range': (60, 85), 
            'burst_window': 100,
            'sparsity_threshold': 0.3,
            'activity_bands': [5, 6, 7]  # HUMAN bands
        }
    
    def extract_car_signature(self, spikegram, duration):
        """
        Extract car-specific signature patterns
        Based on supervised STDP resonator analysis - cars show periodic patterns
        """
        car_bands = spikegram[self.car_signature_params['activity_bands']]
        
        features = {}
        
        # 1. Periodicity Detection (key car characteristic)
        car_pattern = np.mean(car_bands, axis=0)
        features['periodicity_score'] = self._calculate_periodicity_score(car_pattern)
        
        # 2. Temporal Consistency (cars maintain steady patterns)
        consistency_scores = []
        window_size = self.car_signature_params['periodicity_window']
        n_windows = len(car_pattern) // window_size
        
        for i in range(n_windows - 1):
            window1 = car_pattern[i*window_size:(i+1)*window_size]
            window2 = car_pattern[(i+1)*window_size:(i+2)*window_size]
            if len(window1) > 0 and len(window2) > 0:
                correlation = np.corrcoef(window1, window2)[0, 1]
                consistency_scores.append(correlation if not np.isnan(correlation) else 0)
        
        features['temporal_consistency'] = np.mean(consistency_scores) if consistency_scores else 0
        
        # 3. Frequency Band Dominance (cars dominate 30-48 Hz)
        car_power = np.sum(car_bands)
        total_power = np.sum(spikegram)
        features['band_dominance'] = car_power / (total_power + 1e-10)
        
        # 4. Activity Pattern Regularity
        features['activity_regularity'] = self._calculate_activity_regularity(car_pattern)
        
        # 5. Peak Counting (regular peaks indicate vehicle passage)
        features['peak_count'] = self._count_significant_peaks(car_pattern)
        
        # 6. Spectral Coherence across car bands
        features['spectral_coherence'] = self._calculate_cross_band_coherence(car_bands)
        
        return features
    
    def extract_human_signature(self, spikegram, duration):
        """
        Extract human-specific signature patterns
        Based on burst detection from EEG attention analysis - humans show sporadic bursts
        """
        human_bands = spikegram[self.human_signature_params['activity_bands']]
        
        features = {}
        
        # 1. Burst Pattern Analysis (key human characteristic)
        human_pattern = np.mean(human_bands, axis=0)
        features['burstiness'] = self._calculate_burstiness_index(human_pattern)
        
        # 2. Sparsity Analysis (human footsteps are sparse)
        features['sparsity'] = self._calculate_sparsity(human_pattern)
        
        # 3. Event Counting (discrete footstep events)
        features['event_count'] = self._count_discrete_events(human_pattern)
        
        # 4. Inter-Event Interval Analysis
        features['iei_variability'] = self._analyze_inter_event_intervals(human_pattern)
        
        # 5. High-frequency Activity Concentration
        features['hf_concentration'] = self._calculate_hf_concentration(human_bands)
        
        # 6. Temporal Clustering (footsteps cluster in time)
        features['temporal_clustering'] = self._calculate_temporal_clustering(human_pattern)
        
        return features
    
    def extract_discriminative_features(self, chunk):
        """
        Extract the most discriminative features for classification
        Based on range STDP and phase detection analysis
        """
        spikegram = chunk['spikes_bands_spectrogram']
        duration = chunk['duration']
        
        features = {}
        
        # Car signature features
        car_features = self.extract_car_signature(spikegram, duration)
        for key, value in car_features.items():
            features[f'car_{key}'] = value
        
        # Human signature features  
        human_features = self.extract_human_signature(spikegram, duration)
        for key, value in human_features.items():
            features[f'human_{key}'] = value
        
        # Cross-pattern discrimination
        features['pattern_separation'] = abs(car_features.get('band_dominance', 0) - 
                                           human_features.get('hf_concentration', 0))
        
        # Activity type classification
        features['activity_type_score'] = self._classify_activity_type(
            car_features, human_features
        )
        
        # Temporal dynamics
        features.update(self._extract_temporal_dynamics(spikegram))
        
        # Signal quality metrics
        features.update(self._extract_signal_quality_metrics(spikegram))
        
        return features
    
    def _calculate_periodicity_score(self, signal):
        """Calculate periodicity using advanced autocorrelation analysis"""
        if len(signal) < 20:
            return 0
        
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Full autocorrelation
        autocorr = correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find significant peaks
        if len(autocorr) > 10:
            # Adaptive threshold based on signal characteristics
            threshold = 0.1 * np.max(autocorr) + 0.05 * np.mean(autocorr)
            peaks, _ = find_peaks(autocorr[1:], height=threshold, distance=10)
            
            if len(peaks) > 0:
                # Score based on peak strength and regularity
                peak_heights = autocorr[peaks + 1]
                regularity = 1.0 / (np.std(np.diff(peaks)) + 1) if len(peaks) > 1 else 0
                strength = np.mean(peak_heights) / (np.max(autocorr) + 1e-10)
                return strength * regularity * len(peaks) / len(autocorr) * 100
        
        return 0
    
    def _calculate_activity_regularity(self, signal):
        """Calculate activity regularity for vehicle detection"""
        if len(signal) == 0:
            return 0
        
        # Find activity periods
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        active_periods = signal > threshold
        
        # Calculate regularity of activity
        if np.sum(active_periods) > 0:
            # Entropy-based regularity measure
            activity_entropy = entropy(np.histogram(signal, bins=10)[0] + 1e-10)
            max_entropy = np.log(10)  # Maximum entropy for 10 bins
            regularity = 1.0 - (activity_entropy / max_entropy)
            return regularity
        
        return 0
    
    def _count_significant_peaks(self, signal):
        """Count significant peaks with adaptive thresholding"""
        if len(signal) == 0:
            return 0
        
        # Adaptive threshold
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        threshold = signal_mean + 1.5 * signal_std
        
        # Find peaks
        peaks, properties = find_peaks(signal, height=threshold, distance=20)
        
        # Filter by prominence
        if len(peaks) > 0:
            prominences = properties.get('peak_heights', []) - signal_mean
            significant_peaks = peaks[prominences > 0.5 * signal_std]
            return len(significant_peaks)
        
        return 0
    
    def _calculate_cross_band_coherence(self, multi_band_signal):
        """Calculate coherence across frequency bands"""
        if len(multi_band_signal) < 2:
            return 0
        
        coherences = []
        for i in range(len(multi_band_signal)):
            for j in range(i+1, len(multi_band_signal)):
                band1, band2 = multi_band_signal[i], multi_band_signal[j]
                if len(band1) > 0 and len(band2) > 0:
                    correlation = np.corrcoef(band1, band2)[0, 1]
                    if not np.isnan(correlation):
                        coherences.append(abs(correlation))
        
        return np.mean(coherences) if coherences else 0
    
    def _calculate_burstiness_index(self, signal):
        """
        Calculate burstiness index for human footstep detection
        Based on neurological burst detection algorithms
        """
        if len(signal) == 0:
            return 0
        
        # Identify bursts using adaptive threshold
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        burst_threshold = signal_mean + 2.0 * signal_std
        
        # Find burst periods
        bursts = signal > burst_threshold
        
        if np.sum(bursts) == 0:
            return 0
        
        # Calculate burst characteristics
        burst_transitions = np.diff(bursts.astype(int))
        burst_starts = np.where(burst_transitions == 1)[0]
        burst_ends = np.where(burst_transitions == -1)[0]
        
        # Handle edge cases
        if len(burst_starts) == 0:
            return 0
        
        if len(burst_ends) == 0 or burst_ends[0] < burst_starts[0]:
            burst_ends = np.concatenate([[len(signal)-1], burst_ends])
        
        if len(burst_starts) > len(burst_ends):
            burst_starts = burst_starts[:len(burst_ends)]
        
        # Calculate burstiness metrics
        burst_durations = burst_ends - burst_starts
        inter_burst_intervals = burst_starts[1:] - burst_ends[:-1] if len(burst_starts) > 1 else []
        
        # Burstiness index based on coefficient of variation
        if len(inter_burst_intervals) > 0:
            mean_ibi = np.mean(inter_burst_intervals)
            std_ibi = np.std(inter_burst_intervals)
            cv = std_ibi / (mean_ibi + 1e-10)
            
            # Normalize burstiness score
            burstiness = (cv - 1) / (cv + 1) if cv > 0 else 0
            return max(0, burstiness)
        
        return 0
    
    def _calculate_sparsity(self, signal):
        """Calculate signal sparsity for human detection"""
        if len(signal) == 0:
            return 0
        
        # Activity ratio
        threshold = np.mean(signal) + np.std(signal)
        active_ratio = np.sum(signal > threshold) / len(signal)
        
        # Sparsity is inverse of activity
        sparsity = 1.0 - active_ratio
        return sparsity
    
    def _count_discrete_events(self, signal):
        """Count discrete events (footsteps)"""
        if len(signal) == 0:
            return 0
        
        # Use local maxima detection with minimum separation
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        peaks, _ = find_peaks(signal, height=threshold, distance=50)  # Minimum 50 samples between footsteps
        
        return len(peaks)
    
    def _analyze_inter_event_intervals(self, signal):
        """Analyze variability in inter-event intervals"""
        events = self._find_events(signal)
        
        if len(events) < 2:
            return 0
        
        intervals = np.diff(events)
        if len(intervals) > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
            return cv
        
        return 0
    
    def _find_events(self, signal):
        """Find discrete events in signal"""
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        peaks, _ = find_peaks(signal, height=threshold, distance=30)
        return peaks
    
    def _calculate_hf_concentration(self, high_freq_bands):
        """Calculate high-frequency activity concentration"""
        total_hf_activity = np.sum(high_freq_bands)
        max_possible = len(high_freq_bands) * np.max(high_freq_bands) if len(high_freq_bands) > 0 else 1
        
        concentration = total_hf_activity / (max_possible + 1e-10)
        return concentration
    
    def _calculate_temporal_clustering(self, signal):
        """Calculate temporal clustering of events"""
        events = self._find_events(signal)
        
        if len(events) < 3:
            return 0
        
        # Calculate clustering coefficient
        intervals = np.diff(events)
        if len(intervals) > 1:
            # Measure of clustering: inverse of interval variance
            clustering = 1.0 / (np.var(intervals) + 1)
            return clustering
        
        return 0
    
    def _classify_activity_type(self, car_features, human_features):
        """Classify activity type based on feature comparison"""
        car_score = (
            car_features.get('periodicity_score', 0) * 0.3 +
            car_features.get('temporal_consistency', 0) * 0.3 +
            car_features.get('band_dominance', 0) * 0.4
        )
        
        human_score = (
            human_features.get('burstiness', 0) * 0.4 +
            human_features.get('sparsity', 0) * 0.3 +
            human_features.get('event_count', 0) / 10.0 * 0.3  # Normalize event count
        )
        
        # Return discrimination score (positive = car-like, negative = human-like)
        return car_score - human_score
    
    def _extract_temporal_dynamics(self, spikegram):
        """Extract temporal dynamics features"""
        features = {}
        
        # Overall activity evolution
        total_activity = np.sum(spikegram, axis=0)
        
        # Activity gradient (how activity changes over time)
        if len(total_activity) > 1:
            gradient = np.gradient(total_activity)
            features['activity_gradient_mean'] = np.mean(gradient)
            features['activity_gradient_std'] = np.std(gradient)
        else:
            features['activity_gradient_mean'] = 0
            features['activity_gradient_std'] = 0
        
        # Activity concentration in time
        if np.sum(total_activity) > 0:
            time_weights = np.arange(len(total_activity))
            features['temporal_centroid'] = np.sum(time_weights * total_activity) / np.sum(total_activity)
        else:
            features['temporal_centroid'] = 0
        
        # Activity spread
        features['temporal_spread'] = np.std(total_activity)
        
        return features
    
    def _extract_signal_quality_metrics(self, spikegram):
        """Extract signal quality and SNR metrics"""
        features = {}
        
        # Signal-to-noise ratio estimation
        signal_bands = spikegram[1:7]  # Exclude extreme bands
        noise_bands = spikegram[[0, 7]]  # Lowest and highest bands
        
        signal_power = np.mean(signal_bands)
        noise_power = np.mean(noise_bands)
        
        features['snr_estimate'] = signal_power / (noise_power + 1e-10)
        
        # Activity distribution across bands
        band_activities = np.sum(spikegram, axis=1)
        total_activity = np.sum(band_activities)
        
        if total_activity > 0:
            features['activity_entropy'] = entropy(band_activities + 1e-10)
        else:
            features['activity_entropy'] = 0
        
        # Dynamic range
        features['dynamic_range'] = np.max(spikegram) - np.min(spikegram)
        
        return features

def extract_advanced_footprint_features(chunks, signal_type):
    """
    Extract advanced footprint features from chunks using sophisticated pattern analysis
    """
    print(f"🔬 ADVANCED FOOTPRINT ANALYSIS: {signal_type.upper()}")
    print("-" * 50)
    
    detector = AdvancedFootprintDetector()
    features_list = []
    
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}")
        
        # Extract discriminative features
        features = detector.extract_discriminative_features(chunk)
        features_list.append(list(features.values()))
        
        # Print sample analysis for first chunk
        if i == 0:
            print(f"   🎯 Sample features extracted: {len(features)}")
            
            # Show key discriminative features
            key_features = ['car_periodicity_score', 'car_band_dominance', 
                          'human_burstiness', 'human_sparsity', 'pattern_separation']
            
            for key in key_features:
                if key in features:
                    print(f"      {key}: {features[key]:.4f}")
    
    feature_matrix = np.array(features_list)
    print(f"✅ Extracted {feature_matrix.shape[0]} samples with {feature_matrix.shape[1]} features")
    
    return feature_matrix

if __name__ == "__main__":
    print("🔬 Advanced Footprint Detector Module")
    print("Implements sophisticated pattern analysis for geophone signals")
    print("Based on EEG, STDP, and neuromorphic analysis approaches") 