#!/usr/bin/env python3
"""
Comprehensive 30-Second Chunk Analyzer and Footprint Detector
Advanced geophone signal classification system optimized for 30-second segments

This system:
1. Analyzes all 30-second chunks with advanced footprint detection
2. Combines FFT (raw data) and SCTN (chunk) features for optimal performance
3. Builds two separate models: Car vs Car_nothing, Human vs Human_nothing
4. Implements pattern detection similar to EEG attention analysis
5. Provides detailed analysis of each chunk's characteristics

Based on best practices from:
- advanced_geophone_footprint_analyzer.py
- EEG mental attention detection methods
- Supervised STDP approaches from old notebooks
"""

import numpy as np
import pandas as pd
import os
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import welch, spectrogram, periodogram, find_peaks
import warnings
warnings.filterwarnings('ignore')

# Frequency bands optimized for geophone analysis
FREQUENCY_BANDS = {
    'LOW_FREQ': (20, 30),      # Background/noise
    'CAR_APPROACH': (30, 34),  # Car approach signature
    'CAR_PEAK': (34, 40),      # Main car frequency (36.74 Hz optimal)
    'CAR_TAIL': (40, 48),      # Car departure signature
    'MID_GAP': (48, 60),       # Transition zone
    'HUMAN_PEAK': (60, 70),    # Primary human footstep frequency (66.16 Hz optimal)
    'HUMAN_TAIL': (70, 85),    # Secondary human harmonics
    'HIGH_FREQ': (85, 100)     # High frequency activity
}

# Optimal frequencies discovered from previous analysis
OPTIMAL_FREQUENCIES = {
    'car': 36.74,      # Hz - strongest car signature
    'human': 66.16,    # Hz - strongest human signature
}

class Comprehensive30sAnalyzer:
    """
    Advanced analyzer for 30-second geophone chunks
    Implements dual FFT+SCTN approach for optimal footprint detection
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.frequency_bands = FREQUENCY_BANDS
        self.optimal_freqs = OPTIMAL_FREQUENCIES
        self.chunk_cache = {}
        
        # Results storage
        self.chunk_analysis = {}
        self.models = {}
        
        print("🔬 COMPREHENSIVE 30-SECOND CHUNK ANALYZER")
        print("=" * 60)
        print("Advanced footprint detection with dual FFT+SCTN approach")
        print("Building optimized models: Car vs Car_nothing | Human vs Human_nothing")
        print()
        
    def load_and_analyze_all_chunks(self, base_dir="/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output_30s"):
        """
        Load and analyze all 30-second chunks from all categories
        """
        print("📦 LOADING AND ANALYZING ALL 30-SECOND CHUNKS")
        print("-" * 50)
        
        categories = ['car', 'car_nothing', 'human', 'human_nothing']
        
        for category in categories:
            print(f"\n🔍 Analyzing {category.upper()} chunks...")
            chunks_dir = os.path.join(base_dir, category)
            
            if not os.path.exists(chunks_dir):
                print(f"❌ Directory not found: {chunks_dir}")
                continue
                
            # Load chunk index
            index_file = os.path.join(chunks_dir, 'chunk_index.pkl')
            if not os.path.exists(index_file):
                print(f"❌ Chunk index not found: {index_file}")
                continue
                
            with open(index_file, 'rb') as f:
                chunk_index = pickle.load(f)
            
            # Load and analyze each chunk
            chunks_data = []
            for i, chunk_file in enumerate(chunk_index['chunk_files']):
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk = pickle.load(f)
                    chunks_data.append(chunk)
                    
                    # Quick analysis of this chunk
                    footprint_score = self._analyze_chunk_footprint(chunk, category)
                    print(f"   📊 Chunk {i}: footprint score = {footprint_score:.3f}")
            
            # Store chunks for this category
            self.chunk_cache[category] = chunks_data
            
            print(f"   ✅ Loaded {len(chunks_data)} chunks for {category}")
            
            # Detailed analysis
            self._perform_detailed_chunk_analysis(category, chunks_data)
        
        return self.chunk_cache
    
    def _analyze_chunk_footprint(self, chunk, category):
        """
        Quick footprint analysis of a single chunk
        """
        spikegram = chunk['spikes_bands_spectrogram']
        
        # Determine signal type
        signal_type = category.replace('_nothing', '')
        is_nothing = '_nothing' in category
        
        if signal_type == 'car':
            # Car: focus on bands 2-3 (34-48 Hz range)
            target_bands = spikegram[2:4]  # CAR_PEAK and CAR_TAIL
            target_activity = np.sum(target_bands)
            
            # Periodicity check (cars show regular patterns)
            temporal_pattern = np.mean(target_bands, axis=0)
            periodicity = self._calculate_periodicity(temporal_pattern)
            
            # Energy-based scoring
            total_energy = np.sum(spikegram)
            car_dominance = target_activity / (total_energy + 1e-10)
            
            footprint_score = (periodicity + car_dominance) / 2.0
            
        elif signal_type == 'human':
            # Human: focus on bands 5-6 (60-85 Hz range)
            target_bands = spikegram[5:7]  # HUMAN_PEAK and HUMAN_TAIL
            target_activity = np.sum(target_bands)
            
            # Burstiness check (humans show burst patterns)
            temporal_pattern = np.mean(target_bands, axis=0)
            burstiness = self._calculate_burstiness(temporal_pattern)
            
            # Event-based scoring
            total_energy = np.sum(spikegram)
            human_dominance = target_activity / (total_energy + 1e-10)
            
            footprint_score = (burstiness + human_dominance) / 2.0
        else:
            footprint_score = 0.0
        
        return footprint_score
    
    def _perform_detailed_chunk_analysis(self, category, chunks_data):
        """
        Perform detailed statistical analysis of chunks in a category
        """
        print(f"\n📈 DETAILED ANALYSIS: {category.upper()}")
        print("-" * 30)
        
        footprint_scores = []
        activity_levels = []
        pattern_strengths = []
        
        for chunk in chunks_data:
            spikegram = chunk['spikes_bands_spectrogram']
            
            # Calculate various metrics
            footprint_score = self._analyze_chunk_footprint(chunk, category)
            activity_level = np.sum(spikegram)
            
            # Pattern strength analysis
            signal_type = category.replace('_nothing', '')
            if signal_type == 'car':
                car_bands = spikegram[2:4]
                pattern_strength = np.mean(car_bands)
            elif signal_type == 'human':
                human_bands = spikegram[5:7]
                pattern_strength = np.mean(human_bands)
            else:
                pattern_strength = 0
            
            footprint_scores.append(footprint_score)
            activity_levels.append(activity_level)
            pattern_strengths.append(pattern_strength)
        
        # Store analysis results
        self.chunk_analysis[category] = {
            'num_chunks': len(chunks_data),
            'footprint_scores': footprint_scores,
            'activity_levels': activity_levels,
            'pattern_strengths': pattern_strengths,
            'mean_footprint': np.mean(footprint_scores),
            'std_footprint': np.std(footprint_scores),
            'mean_activity': np.mean(activity_levels),
            'mean_pattern': np.mean(pattern_strengths)
        }
        
        # Print summary
        print(f"   📊 Chunks: {len(chunks_data)}")
        print(f"   🎯 Avg footprint score: {np.mean(footprint_scores):.3f} ± {np.std(footprint_scores):.3f}")
        print(f"   ⚡ Avg activity level: {np.mean(activity_levels):,.0f}")
        print(f"   💪 Avg pattern strength: {np.mean(pattern_strengths):.3f}")
        
    def extract_comprehensive_features(self, chunks_data, category):
        """
        Extract comprehensive features combining multiple approaches
        """
        print(f"🔬 Extracting comprehensive features for {category}")
        
        features_list = []
        signal_type = category.replace('_nothing', '')
        is_nothing = '_nothing' in category
        
        for chunk_idx, chunk in enumerate(chunks_data):
            # 1. SCTN-based features from spikegram
            sctn_features = self._extract_sctn_features(chunk, signal_type)
            
            # 2. Footprint-specific features  
            footprint_features = self._extract_footprint_features(chunk, signal_type)
            
            # 3. Temporal pattern features
            temporal_features = self._extract_temporal_features(chunk, signal_type)
            
            # 4. Resonator-based features
            resonator_features = self._extract_resonator_features(chunk)
            
            # Combine all features
            combined_features = np.concatenate([
                sctn_features,
                footprint_features,
                temporal_features,
                resonator_features
            ])
            
            features_list.append(combined_features)
        
        return np.array(features_list)
    
    def _extract_sctn_features(self, chunk, signal_type):
        """
        Extract SCTN-based features from spikegram
        Based on best practices from advanced_geophone_footprint_analyzer.py
        """
        spikegram = chunk['spikes_bands_spectrogram']
        n_bands, n_time_bins = spikegram.shape
        
        features = []
        
        # Band-wise statistical features
        for band_idx in range(n_bands):
            band_data = spikegram[band_idx]
            
            features.extend([
                np.mean(band_data),
                np.std(band_data),
                np.max(band_data),
                np.min(band_data),
                skew(band_data) if len(band_data) > 0 else 0,
                kurtosis(band_data) if len(band_data) > 0 else 0,
                np.sum(band_data > 0) / len(band_data),  # Activity ratio
                np.percentile(band_data, 75),
                np.percentile(band_data, 95)
            ])
        
        # Cross-band relationships
        if signal_type == 'car':
            # Car-specific patterns (bands 1-4: 30-48 Hz)
            car_bands = spikegram[1:5]
            car_strength = np.mean(car_bands)
            car_consistency = np.std([np.corrcoef(car_bands[i], np.mean(car_bands, axis=0))[0,1] 
                                    for i in range(len(car_bands)) if not np.isnan(np.corrcoef(car_bands[i], np.mean(car_bands, axis=0))[0,1])])
            
            features.extend([
                car_strength,
                car_consistency if not np.isnan(car_consistency) else 0,
                np.sum(car_bands) / (np.sum(spikegram) + 1e-10)  # Car dominance
            ])
            
        elif signal_type == 'human':
            # Human-specific patterns (bands 5-7: 60-85 Hz)
            human_bands = spikegram[5:8]
            human_strength = np.mean(human_bands)
            human_burstiness = self._calculate_burstiness(np.mean(human_bands, axis=0))
            
            features.extend([
                human_strength,
                human_burstiness,
                np.sum(human_bands) / (np.sum(spikegram) + 1e-10)  # Human dominance
            ])
        
        return np.array(features)
    
    def _extract_footprint_features(self, chunk, signal_type):
        """
        Extract signal-specific footprint features
        """
        spikegram = chunk['spikes_bands_spectrogram']
        features = []
        
        if signal_type == 'car':
            # Car footprint: periodic patterns at 36.74 Hz (band 2)
            car_optimal_band = spikegram[2]  # CAR_PEAK band
            
            # Periodicity detection
            periodicity = self._calculate_periodicity(car_optimal_band)
            
            # Consistency (steady signal)
            consistency = self._calculate_consistency(car_optimal_band)
            
            # Energy concentration
            car_energy = np.sum(spikegram[1:5])  # All car bands
            total_energy = np.sum(spikegram)
            energy_concentration = car_energy / (total_energy + 1e-10)
            
            features.extend([
                periodicity,
                consistency,
                energy_concentration,
                np.mean(car_optimal_band),
                np.max(car_optimal_band),
                np.std(car_optimal_band)
            ])
            
        elif signal_type == 'human':
            # Human footprint: burst patterns at 66.16 Hz (band 5)
            human_optimal_band = spikegram[5]  # HUMAN_PEAK band
            
            # Burstiness detection
            burstiness = self._calculate_burstiness(human_optimal_band)
            
            # Event detection
            event_count = self._count_footstep_events(human_optimal_band)
            
            # Energy concentration
            human_energy = np.sum(spikegram[5:8])  # All human bands
            total_energy = np.sum(spikegram)
            energy_concentration = human_energy / (total_energy + 1e-10)
            
            features.extend([
                burstiness,
                event_count,
                energy_concentration,
                np.mean(human_optimal_band),
                np.max(human_optimal_band),
                np.std(human_optimal_band)
            ])
        
        return np.array(features)
    
    def _extract_temporal_features(self, chunk, signal_type):
        """
        Extract temporal pattern features
        """
        spikegram = chunk['spikes_bands_spectrogram']
        
        # Overall temporal activity
        total_activity = np.sum(spikegram, axis=0)
        
        # Peak detection
        peaks = find_peaks(total_activity, height=np.mean(total_activity))[0]
        
        # Temporal regularity
        temporal_regularity = self._calculate_temporal_regularity(total_activity)
        
        features = [
            len(peaks),
            np.std(total_activity),
            np.max(total_activity),
            temporal_regularity,
            np.mean(total_activity),
            skew(total_activity) if len(total_activity) > 0 else 0,
            kurtosis(total_activity) if len(total_activity) > 0 else 0
        ]
        
        return np.array(features)
    
    def _extract_resonator_features(self, chunk):
        """
        Extract features from raw resonator outputs
        """
        resonator_outputs = chunk['resonator_outputs']
        clk_freq = 153600  # Standard clock frequency
        duration = chunk['duration']
        
        features = []
        
        if clk_freq in resonator_outputs:
            spike_arrays = resonator_outputs[clk_freq]
            
            # Features from first 8 resonators
            for resonator_idx in range(min(len(spike_arrays), 8)):
                spike_times = spike_arrays[resonator_idx]
                
                if len(spike_times) > 0:
                    # Spike rate
                    spike_rate = len(spike_times) / duration
                    
                    # Inter-spike interval statistics
                    if len(spike_times) > 1:
                        isis = np.diff(spike_times) / clk_freq  # Convert to seconds
                        isi_mean = np.mean(isis)
                        isi_std = np.std(isis)
                        isi_cv = isi_std / (isi_mean + 1e-10)
                    else:
                        isi_mean = isi_std = isi_cv = 0
                    
                    features.extend([spike_rate, isi_mean, isi_std, isi_cv])
                else:
                    features.extend([0, 0, 0, 0])
        else:
            # No resonator data available
            features.extend([0] * 32)  # 8 resonators × 4 features
        
        return np.array(features)
    
    def _calculate_periodicity(self, signal):
        """Calculate periodicity using autocorrelation"""
        if len(signal) < 10:
            return 0
        
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks
        if len(autocorr) > 10:
            peaks = find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))[0]
            return len(peaks) / len(autocorr)
        
        return 0
    
    def _calculate_burstiness(self, signal):
        """Calculate burstiness for human footstep detection"""
        if len(signal) == 0:
            return 0
        
        # Threshold for burst detection
        threshold = np.mean(signal) + 2 * np.std(signal)
        bursts = signal > threshold
        
        # Count burst transitions
        burst_transitions = np.diff(bursts.astype(int))
        burst_starts = np.sum(burst_transitions == 1)
        
        return burst_starts / len(signal) * 100
    
    def _calculate_consistency(self, signal):
        """Calculate signal consistency (inverse of coefficient of variation)"""
        if len(signal) == 0 or np.std(signal) == 0:
            return 0
        
        cv = np.std(signal) / (np.mean(signal) + 1e-10)
        return 1.0 / (1.0 + cv)
    
    def _count_footstep_events(self, signal):
        """Count discrete footstep events"""
        if len(signal) == 0:
            return 0
        
        # Strong threshold for footstep impacts
        threshold = np.mean(signal) + 3 * np.std(signal)
        strong_peaks = signal > threshold
        
        # Group nearby peaks as single events
        events = []
        in_event = False
        
        for i, is_peak in enumerate(strong_peaks):
            if is_peak and not in_event:
                events.append(i)
                in_event = True
            elif not is_peak and in_event:
                in_event = False
        
        return len(events)
    
    def _calculate_temporal_regularity(self, signal):
        """Calculate temporal regularity of patterns"""
        if len(signal) < 10:
            return 0
        
        peaks = find_peaks(signal, height=np.mean(signal))[0]
        
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            return 1.0 / (np.std(intervals) + 1e-10)
        
        return 0
    
    def build_two_models(self):
        """
        Build two separate models:
        1. Car vs Car_nothing
        2. Human vs Human_nothing
        """
        print("\n🎯 BUILDING TWO SPECIALIZED MODELS")
        print("=" * 50)
        
        models = {}
        
        # Model 1: Car vs Car_nothing
        print("\n🚗 Building Car vs Car_nothing model...")
        car_model = self._build_single_model('car', 'car_nothing')
        models['car_model'] = car_model
        
        # Model 2: Human vs Human_nothing
        print("\n👤 Building Human vs Human_nothing model...")
        human_model = self._build_single_model('human', 'human_nothing')
        models['human_model'] = human_model
        
        self.models = models
        return models
    
    def _build_single_model(self, signal_category, nothing_category):
        """
        Build a single specialized model for signal vs nothing detection
        """
        # Extract features for both categories
        print(f"   🔬 Extracting features for {signal_category} and {nothing_category}...")
        
        signal_features = self.extract_comprehensive_features(
            self.chunk_cache[signal_category], signal_category
        )
        nothing_features = self.extract_comprehensive_features(
            self.chunk_cache[nothing_category], nothing_category
        )
        
        # Prepare dataset
        X = np.vstack([signal_features, nothing_features])
        y = np.concatenate([
            np.ones(len(signal_features)),   # Signal = 1
            np.zeros(len(nothing_features))  # Nothing = 0
        ])
        
        print(f"   📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"   📈 Signal samples: {len(signal_features)}")
        print(f"   📉 Nothing samples: {len(nothing_features)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(150, 75), random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        best_score = 0
        best_model = None
        best_name = None
        
        for name, clf in classifiers.items():
            # Train classifier
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = clf.score(X_train_scaled, y_train)
            test_score = clf.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            cv_mean = np.mean(cv_scores)
            
            results[name] = {
                'model': clf,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_mean,
                'cv_std': np.std(cv_scores)
            }
            
            print(f"   🎯 {name}:")
            print(f"      Training: {train_score:.3f}")
            print(f"      Testing: {test_score:.3f}")
            print(f"      CV: {cv_mean:.3f} ± {np.std(cv_scores):.3f}")
            
            # Track best model
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        # Create ensemble of top models
        top_models = sorted(results.items(), key=lambda x: x[1]['test_score'], reverse=True)[:3]
        ensemble_models = [(name, result['model']) for name, result in top_models]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        
        ensemble_score = ensemble.score(X_test_scaled, y_test)
        print(f"   🏆 Ensemble: {ensemble_score:.3f}")
        
        # Final predictions and analysis
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)
        
        print(f"\n   📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Nothing', 'Signal']))
        
        return {
            'scaler': scaler,
            'individual_models': results,
            'ensemble': ensemble,
            'best_individual': {
                'name': best_name,
                'model': best_model,
                'score': best_score
            },
            'ensemble_score': ensemble_score,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_labels': y_test,
            'feature_count': X.shape[1],
            'signal_category': signal_category,
            'nothing_category': nothing_category
        }
    
    def create_comprehensive_report(self):
        """
        Create a comprehensive analysis report
        """
        print("\n📋 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        # Chunk analysis summary
        print("\n📊 CHUNK ANALYSIS SUMMARY:")
        for category, analysis in self.chunk_analysis.items():
            print(f"\n{category.upper()}:")
            print(f"   Chunks: {analysis['num_chunks']}")
            print(f"   Mean footprint score: {analysis['mean_footprint']:.3f}")
            print(f"   Mean activity level: {analysis['mean_activity']:,.0f}")
            print(f"   Mean pattern strength: {analysis['mean_pattern']:.3f}")
        
        # Model performance summary
        if self.models:
            print("\n🎯 MODEL PERFORMANCE SUMMARY:")
            
            for model_name, model_data in self.models.items():
                print(f"\n{model_name.upper()}:")
                print(f"   Best Individual: {model_data['best_individual']['name']} ({model_data['best_individual']['score']:.3f})")
                print(f"   Ensemble Score: {model_data['ensemble_score']:.3f}")
                print(f"   Features Used: {model_data['feature_count']}")
                print(f"   Categories: {model_data['signal_category']} vs {model_data['nothing_category']}")
        
        # Comparative analysis
        print("\n🔍 COMPARATIVE ANALYSIS:")
        
        # Car vs Car_nothing comparison
        if 'car' in self.chunk_analysis and 'car_nothing' in self.chunk_analysis:
            car_signal = self.chunk_analysis['car']['mean_footprint']
            car_nothing = self.chunk_analysis['car_nothing']['mean_footprint']
            car_separation = car_signal / (car_nothing + 1e-10)
            print(f"   🚗 Car signal separation: {car_separation:.2f}x stronger than car_nothing")
        
        # Human vs Human_nothing comparison
        if 'human' in self.chunk_analysis and 'human_nothing' in self.chunk_analysis:
            human_signal = self.chunk_analysis['human']['mean_footprint']
            human_nothing = self.chunk_analysis['human_nothing']['mean_footprint']
            human_separation = human_signal / (human_nothing + 1e-10)
            print(f"   👤 Human signal separation: {human_separation:.2f}x stronger than human_nothing")
        
        # Save detailed report
        report_file = "comprehensive_30s_analysis_report.pkl"
        report_data = {
            'chunk_analysis': self.chunk_analysis,
            'models': self.models,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(report_file, 'wb') as f:
            pickle.dump(report_data, f)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        return report_data

def main():
    """
    Main execution function
    """
    print("🚀 STARTING COMPREHENSIVE 30-SECOND CHUNK ANALYSIS")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = Comprehensive30sAnalyzer()
    
    # Step 1: Load and analyze all chunks
    print("\n📦 STEP 1: LOADING AND ANALYZING CHUNKS")
    chunks_cache = analyzer.load_and_analyze_all_chunks()
    
    if not chunks_cache:
        print("❌ No chunks loaded. Check directory paths.")
        return
    
    # Step 2: Build models
    print("\n🎯 STEP 2: BUILDING SPECIALIZED MODELS")
    models = analyzer.build_two_models()
    
    # Step 3: Create comprehensive report
    print("\n📋 STEP 3: GENERATING COMPREHENSIVE REPORT")
    report = analyzer.create_comprehensive_report()
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 50)
    print("✅ All 30-second chunks analyzed")
    print("✅ Two specialized models built")
    print("✅ Comprehensive report generated")
    print("✅ Results saved for future use")
    
    return analyzer, models, report

if __name__ == "__main__":
    analyzer, models, report = main() 