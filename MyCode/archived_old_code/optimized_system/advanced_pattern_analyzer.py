#!/usr/bin/env python3
# Advanced Pattern Analyzer for Geophone Signals

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternAnalyzer:
    def __init__(self):
        self.band_names = [
            'LOW_FREQ', 'CAR_APPROACH', 'CAR_PEAK', 'CAR_TAIL', 
            'MID_GAP', 'HUMAN_PEAK', 'HUMAN_TAIL', 'HIGH_FREQ'
        ]
        
        # Pattern-specific band groups based on discovered patterns
        self.car_bands = [0, 1, 2, 3]    # Bands with strong car activity
        self.human_bands = [4, 5, 6]     # Bands with strong human activity  
        self.background_bands = [7]       # Background/noise bands
        
    def extract_comprehensive_features(self, spikes_bands_spectrogram, duration, signal_type='unknown'):
        cleaned_data = self._preprocess_spectrogram(spikes_bands_spectrogram)
        
        all_features = []
        all_names = []
        
        # Band Statistics (32 features)
        band_features, band_names = self._extract_band_statistics(cleaned_data)
        all_features.extend(band_features)
        all_names.extend(band_names)
        
        # Temporal Features (8 features)
        temporal_features, temporal_names = self._extract_temporal_patterns(cleaned_data, duration)
        all_features.extend(temporal_features)
        all_names.extend(temporal_names)
        
        # Car-specific features (4 features)
        car_features, car_names = self._extract_car_features(cleaned_data)
        all_features.extend(car_features)
        all_names.extend(car_names)
        
        # Human-specific features (4 features)
        human_features, human_names = self._extract_human_features(cleaned_data)
        all_features.extend(human_features)
        all_names.extend(human_names)
        
        # Cross-band features (4 features)
        cross_features, cross_names = self._extract_cross_band_features(cleaned_data)
        all_features.extend(cross_features)
        all_names.extend(cross_names)
        
        # Convert to numpy array with robust handling
        feature_vector = np.array(all_features, dtype=np.float64)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_vector, all_names
        
    def _preprocess_spectrogram(self, spectrogram):
        cleaned = np.copy(spectrogram).astype(np.float64)
        
        for band_idx in range(spectrogram.shape[0]):
            band_data = spectrogram[band_idx]
            
            # Robust outlier removal using IQR
            q25, q75 = np.percentile(band_data, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 3 * iqr
            upper_bound = q75 + 3 * iqr
            
            # Clip outliers
            cleaned[band_idx] = np.clip(band_data, lower_bound, upper_bound)
        
        return cleaned
        
    def _extract_band_statistics(self, spectrogram):
        features = []
        names = []
        
        for i, band_name in enumerate(self.band_names):
            band_data = spectrogram[i]
            
            features.extend([
                np.mean(band_data),
                np.std(band_data),
                np.max(band_data),
                np.sum(band_data > 0) / len(band_data)
            ])
            
            names.extend([
                f'{band_name}_mean',
                f'{band_name}_std', 
                f'{band_name}_max',
                f'{band_name}_activity_ratio'
            ])
        
        return features, names
        
    def _extract_temporal_patterns(self, spectrogram, duration):
        total_activity = np.sum(spectrogram, axis=0)
        
        features = []
        names = []
        
        # Basic temporal statistics
        features.extend([
            np.mean(total_activity),
            np.std(total_activity),
            stats.skew(total_activity),
            stats.kurtosis(total_activity)
        ])
        names.extend(['temp_mean', 'temp_std', 'temp_skew', 'temp_kurtosis'])
        
        # Peak detection
        if len(total_activity) > 10:
            peaks, _ = find_peaks(total_activity, height=np.mean(total_activity))
            features.extend([
                len(peaks),
                len(peaks) / duration,
                np.mean(total_activity[peaks]) if len(peaks) > 0 else 0
            ])
        else:
            features.extend([0, 0, 0])
        names.extend(['temp_peak_count', 'temp_peak_density', 'temp_peak_height'])
        
        # Periodicity using autocorrelation
        if len(total_activity) > 20:
            periodicity = self._calculate_periodicity(total_activity)
            features.append(periodicity)
        else:
            features.append(0)
        names.append('temp_periodicity')
        
        return features, names
        
    def _extract_car_features(self, spectrogram):
        car_activity = np.sum(spectrogram[self.car_bands], axis=0)
        total_activity = np.sum(spectrogram, axis=0)
        
        features = []
        names = []
        
        # Car dominance
        car_total = np.sum(car_activity)
        overall_total = np.sum(total_activity)
        car_dominance = car_total / (overall_total + 1e-10)
        features.append(car_dominance)
        names.append('car_dominance')
        
        # Car consistency (cars should have consistent activity)
        car_consistency = 1 / (1 + np.std(car_activity) / (np.mean(car_activity) + 1e-10))
        features.append(car_consistency)
        names.append('car_consistency')
        
        # Car periodicity
        if len(car_activity) > 20:
            car_periodicity = self._calculate_periodicity(car_activity)
            features.append(car_periodicity)
        else:
            features.append(0)
        names.append('car_periodicity')
        
        # Low-frequency dominance (observed pattern)
        low_freq_ratio = np.sum(spectrogram[0]) / (overall_total + 1e-10)
        features.append(low_freq_ratio)
        names.append('car_low_freq_ratio')
        
        return features, names
        
    def _extract_human_features(self, spectrogram):
        human_activity = np.sum(spectrogram[self.human_bands], axis=0)
        total_activity = np.sum(spectrogram, axis=0)
        
        features = []
        names = []
        
        # Human dominance
        human_total = np.sum(human_activity)
        overall_total = np.sum(total_activity)
        human_dominance = human_total / (overall_total + 1e-10)
        features.append(human_dominance)
        names.append('human_dominance')
        
        # Burst characteristics
        if len(human_activity) > 10:
            burst_score = self._calculate_burst_score(human_activity)
            features.append(burst_score)
        else:
            features.append(0)
        names.append('human_burst_score')
        
        # Variability (humans more variable than cars)
        human_variability = np.std(human_activity) / (np.mean(human_activity) + 1e-10)
        features.append(human_variability)
        names.append('human_variability')
        
        # High-frequency preference
        high_freq_ratio = np.sum(spectrogram[self.human_bands]) / (overall_total + 1e-10)
        features.append(high_freq_ratio)
        names.append('human_high_freq_ratio')
        
        return features, names
        
    def _extract_cross_band_features(self, spectrogram):
        features = []
        names = []
        
        # Band activity ratios
        band_totals = np.sum(spectrogram, axis=1)
        total_sum = np.sum(band_totals) + 1e-10
        
        # Key ratios
        low_vs_high = (band_totals[0] + band_totals[1]) / (band_totals[6] + band_totals[7] + 1e-10)
        car_vs_human = np.sum(band_totals[self.car_bands]) / (np.sum(band_totals[self.human_bands]) + 1e-10)
        
        features.extend([low_vs_high, car_vs_human])
        names.extend(['ratio_low_vs_high', 'ratio_car_vs_human'])
        
        # Spectral characteristics
        band_indices = np.arange(len(band_totals))
        centroid = np.sum(band_indices * band_totals) / total_sum
        spread = np.sqrt(np.sum(((band_indices - centroid) ** 2) * band_totals) / total_sum)
        
        features.extend([centroid, spread])
        names.extend(['spectral_centroid', 'spectral_spread'])
        
        return features, names
        
    def _calculate_periodicity(self, signal_data):
        if len(signal_data) < 20:
            return 0
        
        signal_norm = signal_data - np.mean(signal_data)
        if np.std(signal_norm) == 0:
            return 0
        
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 10:
            peaks, _ = find_peaks(autocorr[1:], height=0.1 * autocorr[0])
            if len(peaks) > 0:
                return np.max(autocorr[peaks + 1]) / autocorr[0]
        return 0
        
    def _calculate_burst_score(self, signal_data):
        if len(signal_data) < 10:
            return 0
        
        threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
        burst_points = signal_data > threshold
        
        if np.sum(burst_points) == 0:
            return 0
        
        burst_count = self._count_events(burst_points)
        burst_intensity = np.mean(signal_data[burst_points]) / (np.mean(signal_data) + 1e-10)
        burst_duration = np.mean(burst_points)
        
        return burst_count * burst_intensity * burst_duration
        
    def _count_events(self, binary_signal):
        if len(binary_signal) == 0:
            return 0
        
        events = 0
        in_event = False
        
        for active in binary_signal:
            if active and not in_event:
                events += 1
                in_event = True
            elif not active:
                in_event = False
        
        return events

    def extract_segments_with_features(self, spikes_bands_spectrogram, duration, signal_type='unknown', 
                                     segment_duration=10, overlap=0.5):
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        samples_per_segment = int(segment_duration * 100)  # 100 samples per second
        
        if n_time_bins < samples_per_segment:
            # If data is too short, return single segment
            features, names = self.extract_comprehensive_features(
                spikes_bands_spectrogram, duration, signal_type
            )
            return [features], names
        
        # Extract overlapping segments
        step_size = int(samples_per_segment * (1 - overlap))
        segments = []
        
        for start_idx in range(0, n_time_bins - samples_per_segment + 1, step_size):
            end_idx = start_idx + samples_per_segment
            segment_data = spikes_bands_spectrogram[:, start_idx:end_idx]
            
            # Extract comprehensive features for this segment
            features, names = self.extract_comprehensive_features(
                segment_data, segment_duration, signal_type
            )
            segments.append(features)
        
        return segments, names

if __name__ == '__main__':
    print('Advanced Pattern Analyzer created successfully')
    analyzer = AdvancedPatternAnalyzer()
    print('Feature extraction methods:', [m for m in dir(analyzer) if m.startswith('_extract')])
