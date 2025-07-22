#!/usr/bin/env python3
"""
Advanced Footprint Detector
Sophisticated pattern analysis for geophone signals
"""

import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.stats import entropy

class FootprintDetector:
    """Advanced pattern detector for geophone analysis"""
    
    def __init__(self):
        self.car_bands = [1, 2, 3, 4]  # 30-48 Hz
        self.human_bands = [5, 6, 7]  # 60-85 Hz
    
    def extract_car_features(self, spikegram):
        """Extract car-specific features"""
        car_data = spikegram[self.car_bands]
        car_pattern = np.mean(car_data, axis=0)
        
        features = {
            'periodicity': self._calculate_periodicity(car_pattern),
            'consistency': self._calculate_consistency(car_data),
            'dominance': np.sum(car_data) / (np.sum(spikegram) + 1e-10),
            'peak_count': self._count_peaks(car_pattern)
        }
        return features
    
    def extract_human_features(self, spikegram):
        """Extract human-specific features"""
        human_data = spikegram[self.human_bands]
        human_pattern = np.mean(human_data, axis=0)
        
        features = {
            'burstiness': self._calculate_burstiness(human_pattern),
            'sparsity': self._calculate_sparsity(human_pattern),
            'event_count': self._count_events(human_pattern),
            'concentration': np.sum(human_data) / (np.sum(spikegram) + 1e-10)
        }
        return features
    
    def _calculate_periodicity(self, signal):
        """Calculate periodicity score"""
        if len(signal) < 20:
            return 0
        
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        autocorr = correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 10:
            threshold = 0.1 * np.max(autocorr)
            peaks, _ = find_peaks(autocorr[1:], height=threshold)
            return len(peaks) / len(autocorr) * 100
        return 0
    
    def _calculate_consistency(self, multi_band_data):
        """Calculate temporal consistency"""
        if len(multi_band_data) < 2:
            return 0
        
        correlations = []
        for i in range(len(multi_band_data)):
            for j in range(i+1, len(multi_band_data)):
                corr = np.corrcoef(multi_band_data[i], multi_band_data[j])[0,1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        return np.mean(correlations) if correlations else 0
    
    def _count_peaks(self, signal):
        """Count significant peaks"""
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        peaks, _ = find_peaks(signal, height=threshold, distance=20)
        return len(peaks)
    
    def _calculate_burstiness(self, signal):
        """Calculate burstiness index"""
        if len(signal) == 0:
            return 0
        
        threshold = np.mean(signal) + 2 * np.std(signal)
        bursts = signal > threshold
        
        if np.sum(bursts) == 0:
            return 0
        
        # Find burst intervals
        burst_changes = np.diff(bursts.astype(int))
        burst_starts = np.where(burst_changes == 1)[0]
        
        if len(burst_starts) < 2:
            return 0
        
        intervals = np.diff(burst_starts)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        return (cv - 1) / (cv + 1) if cv > 0 else 0
    
    def _calculate_sparsity(self, signal):
        """Calculate signal sparsity"""
        threshold = np.mean(signal) + np.std(signal)
        active_ratio = np.sum(signal > threshold) / len(signal)
        return 1.0 - active_ratio
    
    def _count_events(self, signal):
        """Count discrete events"""
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        peaks, _ = find_peaks(signal, height=threshold, distance=50)
        return len(peaks) 