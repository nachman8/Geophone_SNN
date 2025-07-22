#!/usr/bin/env python3
"""
Optimized Pattern Analysis for Geophone Signals
Advanced feature extraction based on identified spikegram patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

class PatternAnalyzer:
    """
    Advanced pattern analysis for geophone signals based on spikegram characteristics
    """
    
    def __init__(self):
        self.band_names = [
            'LOW_FREQ', 'CAR_APPROACH', 'CAR_PEAK', 'CAR_TAIL', 
            'MID_GAP', 'HUMAN_PEAK', 'HUMAN_TAIL', 'HIGH_FREQ'
        ]
        self.band_ranges = [
            (20, 30), (30, 34), (34, 40), (40, 48),
            (48, 60), (60, 70), (70, 80), (90, 100)
        ]
        
        # Pattern-specific band groups
        self.car_bands = [0, 1, 2, 3]  # LOW_FREQ, CAR_APPROACH, CAR_PEAK, CAR_TAIL
        self.human_bands = [4, 5, 6]   # MID_GAP, HUMAN_PEAK, HUMAN_TAIL
        self.background_bands = [7]     # HIGH_FREQ
        
    def extract_comprehensive_features(self, spikes_bands_spectrogram, duration, signal_type='unknown'):
        """
        Extract comprehensive features optimized for car vs human vs nothing classification
        """
        n_bands, n_time_bins = spikes_bands_spectrogram.shape
        
        # Preprocessing: Clean and normalize
        cleaned_spectrogram = self._preprocess_spectrogram(spikes_bands_spectrogram)
        
        # Feature categories
        features = {}
        
        # 1. Band-wise Activity Features (8 features)
        features.update(self._extract_band_activity_features(cleaned_spectrogram))
        
        # 2. Temporal Pattern Features (12 features)
        features.update(self._extract_temporal_features(cleaned_spectrogram, duration))
        
        # 3. Pattern-Specific Features (16 features)
        features.update(self._extract_pattern_specific_features(cleaned_spectrogram, signal_type))
        
        # 4. Cross-Band Interaction Features (8 features)
        features.update(self._extract_cross_band_features(cleaned_spectrogram))
        
        # 5. Statistical Distribution Features (12 features)
        features.update(self._extract_statistical_features(cleaned_spectrogram))
        
        # 6. Event Detection Features (8 features)
        features.update(self._extract_event_features(cleaned_spectrogram))
        
        # Convert to feature vector
        feature_vector = []
        feature_names = []
        
        for category, category_features in features.items():
            for feature_name, feature_value in category_features.items():
                feature_vector.append(feature_value)
                feature_names.append(f"{category}_{feature_name}")
        
        return np.array(feature_vector), feature_names
