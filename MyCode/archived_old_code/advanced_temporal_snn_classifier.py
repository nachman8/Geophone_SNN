#!/usr/bin/env python3
"""
Advanced Temporal SNN Classifier with Footprint Recognition
Based on research insights for direct spikegram pattern extraction

Key Features:
1. Direct Spikegram Pattern Recognition - extracts temporal footprints
2. Frequency-Band Specific Analysis - car bands (30-48Hz) vs human bands (48-80Hz) 
3. Temporal Pattern Classification - uses autocorrelation and burst detection
4. Binary Classifiers - separate car vs car_nothing and human vs human_nothing
5. Template-based Matching - similarity scoring for classification

This approach focuses on the actual "footprints" you see in spikegrams!
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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class FootprintPatternClassifier:
    """
    Direct spikegram footprint pattern classifier
    Extracts and matches characteristic patterns from spikegrams
    """
    
    def __init__(self):
        # Frequency band definitions based on user observations
        self.band_names = [
            'LOW_FREQ', 'CAR_APPROACH', 'CAR_PEAK', 'CAR_TAIL', 
            'MID_GAP', 'HUMAN_PEAK', 'HUMAN_TAIL', 'HIGH_FREQ'
        ]
        
        # Key frequency bands for each signal type
        self.car_bands = [1, 2, 3]      # CAR_APPROACH, CAR_PEAK, CAR_TAIL (30-48 Hz)
        self.human_bands = [4, 5, 6]    # MID_GAP, HUMAN_PEAK, HUMAN_TAIL (48-80 Hz)
        
        # Store learned patterns
        self.signal_templates = {}
        self.classification_thresholds = {}
        
        print(f"👣 FootprintPatternClassifier initialized")
        print(f"   🚗 Car footprint bands: {self.car_bands} (30-48 Hz)")
        print(f"   👤 Human footprint bands: {self.human_bands} (48-80 Hz)")
    
    def extract_footprint_pattern(self, spikegram, signal_type):
        """
        Extract the characteristic footprint pattern from a spikegram
        This is the key function that identifies the patterns you see visually!
        """
        n_bands, n_time_bins = spikegram.shape
        
        # Select relevant frequency bands for this signal type
        if signal_type.startswith('car'):
            target_bands = self.car_bands
            pattern_name = 'car'
        elif signal_type.startswith('human'):
            target_bands = self.human_bands
            pattern_name = 'human'
        else:
            target_bands = list(range(n_bands))
            pattern_name = 'generic'
        
        print(f"   🔍 Extracting {pattern_name} footprint from {n_bands}x{n_time_bins} spikegram")
        
        # Extract temporal pattern from target frequency bands
        target_band_data = spikegram[target_bands, :]
        temporal_footprint = np.mean(target_band_data, axis=0)
        
        # Calculate footprint characteristics
        footprint = {
            'temporal_pattern': temporal_footprint,
            'pattern_type': pattern_name,
            'duration': n_time_bins,
            'signal_type': signal_type
        }
        
        # 1. Activity strength - total energy in target bands
        footprint['activity_strength'] = np.sum(temporal_footprint)
        
        # 2. Pattern consistency (car = steady, human = variable)
        if np.mean(temporal_footprint) > 0:
            footprint['consistency'] = 1.0 / (1.0 + np.std(temporal_footprint) / np.mean(temporal_footprint))
        else:
            footprint['consistency'] = 0.0
        
        # 3. Periodicity detection (cars show regular patterns)
        footprint['periodicity'] = self._detect_periodicity(temporal_footprint)
        
        # 4. Burst detection (humans show burst patterns)
        footprint['burst_score'] = self._detect_bursts(temporal_footprint)
        
        # 5. Frequency band energy distribution
        all_band_energies = np.sum(spikegram, axis=1)
        if np.sum(all_band_energies) > 0:
            footprint['frequency_profile'] = all_band_energies / np.sum(all_band_energies)
        else:
            footprint['frequency_profile'] = np.zeros(n_bands)
        
        # 6. Target band dominance (how much energy is in target vs other bands)
        target_energy = np.sum(all_band_energies[target_bands])
        total_energy = np.sum(all_band_energies)
        footprint['target_dominance'] = target_energy / (total_energy + 1e-10)
        
        # 7. Temporal concentration (how spread out activity is)
        footprint['temporal_concentration'] = self._calculate_temporal_concentration(temporal_footprint)
        
        print(f"     📊 Footprint: strength={footprint['activity_strength']:.2f}, "
              f"consistency={footprint['consistency']:.3f}, "
              f"periodicity={footprint['periodicity']:.3f}, "
              f"bursts={footprint['burst_score']:.1f}")
        
        return footprint
    
    def _detect_periodicity(self, signal):
        """
        Detect periodic patterns using autocorrelation
        Cars show regular periodic patterns, humans don't
        """
        if len(signal) < 20:
            return 0.0
        
        # Normalize signal
        signal_norm = signal - np.mean(signal)
        if np.std(signal_norm) == 0:
            return 0.0
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        
        if len(autocorr) < 10:
            return 0.0
        
        # Look for peaks in autocorrelation (excluding lag 0)
        max_lag = min(100, len(autocorr)//3)
        if max_lag > 1:
            # Find the maximum autocorrelation value (excluding lag 0)
            peak_autocorr = np.max(autocorr[1:max_lag])
            # Normalize by autocorrelation at lag 0
            periodicity_strength = peak_autocorr / (autocorr[0] + 1e-10)
            return max(0.0, periodicity_strength)
        
        return 0.0
    
    def _detect_bursts(self, signal):
        """
        Detect burst-like activity patterns
        Humans show burst patterns, cars show steady patterns
        """
        if len(signal) < 10:
            return 0.0
        
        # Define burst threshold as mean + 1.5 * std
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        burst_points = signal > threshold
        
        # Count burst events (transitions from below to above threshold)
        burst_count = 0
        in_burst = False
        
        for is_above_threshold in burst_points:
            if is_above_threshold and not in_burst:
                burst_count += 1
                in_burst = True
            elif not is_above_threshold:
                in_burst = False
        
        # Normalize by signal duration
        burst_rate = burst_count / len(signal) * 100  # bursts per 100 time steps
        return burst_rate
    
    def _calculate_temporal_concentration(self, signal):
        """
        Calculate how concentrated activity is in time
        Higher values = more concentrated (spiky), lower = more spread out
        """
        if len(signal) == 0 or np.sum(signal) == 0:
            return 0.0
        
        # Normalize to probability distribution
        total_activity = np.sum(signal)
        probabilities = signal / total_activity
        
        # Calculate entropy
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        if len(probabilities) <= 1:
            return 1.0
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(probabilities))
        
        # Convert to concentration (1 - normalized entropy)
        concentration = 1.0 - (entropy / max_entropy)
        return max(0.0, concentration)
    
    def train_binary_classifier(self, positive_spikegrams, negative_spikegrams, signal_type):
        """
        Train binary classifier for signal_type vs signal_type_nothing
        """
        print(f"\n🎯 Training {signal_type} classifier")
        print(f"   ✅ Positive samples: {len(positive_spikegrams)}")
        print(f"   ❌ Negative samples: {len(negative_spikegrams)}")
        
        # Extract footprints from all training data
        positive_footprints = []
        negative_footprints = []
        
        print("   📊 Extracting positive footprints...")
        for i, spikegram in enumerate(positive_spikegrams):
            footprint = self.extract_footprint_pattern(spikegram, signal_type)
            positive_footprints.append(footprint)
            if i < 3:
                print(f"     Sample {i+1}: {footprint['activity_strength']:.2f} strength")
        
        print("   📊 Extracting negative footprints...")
        for i, spikegram in enumerate(negative_spikegrams):
            footprint = self.extract_footprint_pattern(spikegram, f"{signal_type}_nothing")
            negative_footprints.append(footprint)
            if i < 3:
                print(f"     Sample {i+1}: {footprint['activity_strength']:.2f} strength")
        
        # Create template from positive samples
        template = self._create_template(positive_footprints, signal_type)
        self.signal_templates[signal_type] = template
        
        # Calculate similarity scores for all training samples
        positive_scores = []
        negative_scores = []
        
        for footprint in positive_footprints:
            score = self._calculate_similarity(footprint, template)
            positive_scores.append(score)
        
        for footprint in negative_footprints:
            score = self._calculate_similarity(footprint, template)
            negative_scores.append(score)
        
        # Find optimal classification threshold
        threshold, accuracy = self._find_optimal_threshold(positive_scores, negative_scores)
        self.classification_thresholds[signal_type] = threshold
        
        print(f"   ✅ Training completed!")
        print(f"   📊 Training accuracy: {accuracy:.1%}")
        print(f"   🎯 Classification threshold: {threshold:.3f}")
        print(f"   📈 Positive scores: {np.mean(positive_scores):.3f} ± {np.std(positive_scores):.3f}")
        print(f"   📉 Negative scores: {np.mean(negative_scores):.3f} ± {np.std(negative_scores):.3f}")
        
        return accuracy
    
    def _create_template(self, footprints, signal_type):
        """
        Create template pattern from positive training footprints
        """
        print(f"   🏗️  Creating template for {signal_type}")
        
        template = {
            'signal_type': signal_type,
            'n_samples': len(footprints)
        }
        
        # Average numerical features
        numerical_features = [
            'activity_strength', 'consistency', 'periodicity', 
            'burst_score', 'target_dominance', 'temporal_concentration'
        ]
        
        for feature in numerical_features:
            values = [fp[feature] for fp in footprints]
            template[feature] = np.mean(values)
            template[f"{feature}_std"] = np.std(values)
        
        # Average frequency profiles
        frequency_profiles = [fp['frequency_profile'] for fp in footprints]
        template['frequency_profile'] = np.mean(frequency_profiles, axis=0)
        
        # Average temporal patterns (normalize lengths first)
        temporal_patterns = [fp['temporal_pattern'] for fp in footprints]
        max_length = max(len(tp) for tp in temporal_patterns)
        
        normalized_patterns = []
        for tp in temporal_patterns:
            if len(tp) < max_length:
                # Pad with zeros
                padded = np.pad(tp, (0, max_length - len(tp)), 'constant', constant_values=0)
                normalized_patterns.append(padded)
            else:
                normalized_patterns.append(tp[:max_length])
        
        template['temporal_pattern'] = np.mean(normalized_patterns, axis=0)
        
        print(f"     📊 Template features:")
        print(f"       Strength: {template['activity_strength']:.2f}")
        print(f"       Consistency: {template['consistency']:.3f}")
        print(f"       Periodicity: {template['periodicity']:.3f}")
        print(f"       Burst score: {template['burst_score']:.1f}")
        print(f"       Target dominance: {template['target_dominance']:.3f}")
        
        return template
    
    def _calculate_similarity(self, footprint, template):
        """
        Calculate similarity between footprint and template
        Returns score between 0 and 1 (higher = more similar)
        """
        score = 0.0
        
        # Feature weights (adjust based on importance)
        weights = {
            'activity_strength': 0.20,
            'consistency': 0.15,
            'periodicity': 0.15,
            'burst_score': 0.15,
            'target_dominance': 0.20,
            'temporal_concentration': 0.10,
            'temporal_pattern': 0.05
        }
        
        # 1. Activity strength similarity
        fp_strength = footprint['activity_strength']
        template_strength = template['activity_strength']
        if template_strength > 0:
            strength_ratio = min(fp_strength, template_strength) / max(fp_strength, template_strength)
        else:
            strength_ratio = 1.0 if fp_strength == 0 else 0.0
        score += weights['activity_strength'] * strength_ratio
        
        # 2. Consistency similarity
        fp_consistency = footprint['consistency']
        template_consistency = template['consistency']
        consistency_diff = abs(fp_consistency - template_consistency)
        consistency_sim = np.exp(-consistency_diff * 5)  # Gaussian-like similarity
        score += weights['consistency'] * consistency_sim
        
        # 3. Periodicity similarity  
        fp_periodicity = footprint['periodicity']
        template_periodicity = template['periodicity']
        periodicity_diff = abs(fp_periodicity - template_periodicity)
        periodicity_sim = np.exp(-periodicity_diff * 3)
        score += weights['periodicity'] * periodicity_sim
        
        # 4. Burst score similarity
        fp_burst = footprint['burst_score']
        template_burst = template['burst_score']
        burst_diff = abs(fp_burst - template_burst) / 10.0  # Normalize
        burst_sim = np.exp(-burst_diff)
        score += weights['burst_score'] * burst_sim
        
        # 5. Target dominance similarity
        fp_dominance = footprint['target_dominance']
        template_dominance = template['target_dominance']
        dominance_diff = abs(fp_dominance - template_dominance)
        dominance_sim = np.exp(-dominance_diff * 5)
        score += weights['target_dominance'] * dominance_sim
        
        # 6. Temporal concentration similarity
        fp_concentration = footprint['temporal_concentration']
        template_concentration = template['temporal_concentration']
        concentration_diff = abs(fp_concentration - template_concentration)
        concentration_sim = np.exp(-concentration_diff * 3)
        score += weights['temporal_concentration'] * concentration_sim
        
        # 7. Temporal pattern correlation
        fp_temporal = footprint['temporal_pattern']
        template_temporal = template['temporal_pattern']
        
        # Normalize lengths
        min_length = min(len(fp_temporal), len(template_temporal))
        if min_length > 5:  # Only if patterns are long enough
            fp_temporal = fp_temporal[:min_length]
            template_temporal = template_temporal[:min_length]
            
            # Calculate correlation if both have variation
            if np.std(fp_temporal) > 0 and np.std(template_temporal) > 0:
                correlation = np.corrcoef(fp_temporal, template_temporal)[0, 1]
                if not np.isnan(correlation):
                    pattern_sim = max(0.0, correlation)  # Only positive correlation
                    score += weights['temporal_pattern'] * pattern_sim
        
        return min(1.0, max(0.0, score))
    
    def _find_optimal_threshold(self, positive_scores, negative_scores):
        """
        Find optimal classification threshold using training data
        """
        all_scores = positive_scores + negative_scores
        all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)
        
        best_threshold = 0.5
        best_accuracy = 0.0
        
        # Try different thresholds
        for threshold in np.linspace(0.0, 1.0, 101):
            predictions = [1 if score > threshold else 0 for score in all_scores]
            accuracy = sum(p == l for p, l in zip(predictions, all_labels)) / len(all_labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy
    
    def predict(self, spikegram, signal_type):
        """
        Predict if spikegram contains the signal pattern
        Returns confidence score (0-1)
        """
        if signal_type not in self.signal_templates:
            print(f"❌ No trained template for {signal_type}")
            return 0.0
        
        # Extract footprint from input spikegram
        footprint = self.extract_footprint_pattern(spikegram, signal_type)
        
        # Calculate similarity to template
        template = self.signal_templates[signal_type]
        similarity = self._calculate_similarity(footprint, template)
        
        # Convert to probability using threshold
        threshold = self.classification_thresholds[signal_type]
        
        # Sigmoid-like conversion around threshold
        if similarity > threshold:
            # Above threshold: map [threshold, 1] to [0.5, 1]
            confidence = 0.5 + 0.5 * (similarity - threshold) / (1.0 - threshold + 1e-10)
        else:
            # Below threshold: map [0, threshold] to [0, 0.5]
            confidence = 0.5 * similarity / (threshold + 1e-10)
        
        return min(1.0, max(0.0, confidence))

class ChunkedDataLoader:
    """
    Load and manage chunked spikegram data
    """
    
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        self.data_cache = {}
        
        print(f"📁 ChunkedDataLoader initialized")
        print(f"   📂 Directory: {chunks_dir}")
    
    def load_signal_data(self, signal_type):
        """
        Load all chunks for a signal type
        """
        if signal_type in self.data_cache:
            return self.data_cache[signal_type]
        
        signal_dir = self.chunks_dir / signal_type
        
        if not signal_dir.exists():
            print(f"❌ Directory not found: {signal_dir}")
            return []
        
        spikegrams = []
        chunk_files = sorted(signal_dir.glob("chunk_*/chunk_*_data.pkl"))
        
        print(f"📊 Loading {signal_type} data...")
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                if 'spikes_bands_spectrogram' in chunk_data:
                    spikegram = chunk_data['spikes_bands_spectrogram']
                    spikegrams.append(spikegram)
                    
            except Exception as e:
                print(f"   ❌ Error loading {chunk_file}: {e}")
        
        self.data_cache[signal_type] = spikegrams
        print(f"   ✅ Loaded {len(spikegrams)} spikegrams for {signal_type}")
        
        return spikegrams

def evaluate_classifier(classifier, test_positive, test_negative, signal_type):
    """
    Evaluate binary classifier performance
    """
    print(f"\n📋 Evaluating {signal_type} classifier")
    print(f"   🧪 Test samples: {len(test_positive)} positive, {len(test_negative)} negative")
    
    predictions = []
    labels = []
    confidences = []
    
    # Test positive samples
    for i, spikegram in enumerate(test_positive):
        confidence = classifier.predict(spikegram, signal_type)
        prediction = confidence > 0.5
        
        predictions.append(prediction)
        labels.append(True)
        confidences.append(confidence)
        
        if i < 3:
            print(f"   ✅ Positive {i+1}: confidence = {confidence:.3f}")
    
    # Test negative samples
    for i, spikegram in enumerate(test_negative):
        confidence = classifier.predict(spikegram, signal_type)
        prediction = confidence > 0.5
        
        predictions.append(prediction)
        labels.append(False)
        confidences.append(confidence)
        
        if i < 3:
            print(f"   ❌ Negative {i+1}: confidence = {confidence:.3f}")
    
    # Calculate metrics
    tp = sum(p and l for p, l in zip(predictions, labels))  # True positives
    fp = sum(p and not l for p, l in zip(predictions, labels))  # False positives
    tn = sum(not p and not l for p, l in zip(predictions, labels))  # True negatives
    fn = sum(not p and l for p, l in zip(predictions, labels))  # False negatives
    
    accuracy = (tp + tn) / len(predictions)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'confidences': confidences
    }
    
    print(f"   📊 Results:")
    print(f"     Accuracy:  {accuracy:.1%}")
    print(f"     Precision: {precision:.1%}")
    print(f"     Recall:    {recall:.1%}")
    print(f"     F1-Score:  {f1:.1%}")
    print(f"     Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return results

def run_footprint_classification():
    """
    Main function to run the footprint-based classification
    """
    print("🚀 Starting Footprint-Based Spikegram Classification")
    print("=" * 70)
    
    # Initialize system
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    data_loader = ChunkedDataLoader(chunks_dir)
    classifier = FootprintPatternClassifier()
    
    print("\n📁 Loading chunked data...")
    
    # Load all signal data
    car_data = data_loader.load_signal_data('car')
    car_nothing_data = data_loader.load_signal_data('car_nothing')
    human_data = data_loader.load_signal_data('human')
    human_nothing_data = data_loader.load_signal_data('human_nothing')
    
    # Check if we have sufficient data
    if len(car_data) < 2 or len(car_nothing_data) < 2:
        print("❌ Insufficient car data for classification")
        return None
    
    if len(human_data) < 2 or len(human_nothing_data) < 2:
        print("❌ Insufficient human data for classification")
        return None
    
    print(f"\n📊 Dataset summary:")
    print(f"   🚗 Car:         {len(car_data)} spikegrams")
    print(f"   🚫 Car nothing: {len(car_nothing_data)} spikegrams")
    print(f"   👤 Human:       {len(human_data)} spikegrams")
    print(f"   🚫 Human nothing: {len(human_nothing_data)} spikegrams")
    
    # Split data into training and testing sets
    train_ratio = 0.7
    
    # Car data split
    n_car_train = max(1, int(len(car_data) * train_ratio))
    n_car_nothing_train = max(1, int(len(car_nothing_data) * train_ratio))
    
    car_train = car_data[:n_car_train]
    car_test = car_data[n_car_train:]
    car_nothing_train = car_nothing_data[:n_car_nothing_train]
    car_nothing_test = car_nothing_data[n_car_nothing_train:]
    
    # Human data split
    n_human_train = max(1, int(len(human_data) * train_ratio))
    n_human_nothing_train = max(1, int(len(human_nothing_data) * train_ratio))
    
    human_train = human_data[:n_human_train]
    human_test = human_data[n_human_train:]
    human_nothing_train = human_nothing_data[:n_human_nothing_train]
    human_nothing_test = human_nothing_data[n_human_nothing_train:]
    
    print(f"\n🎯 Training set sizes:")
    print(f"   🚗 Car: {len(car_train)} positive, {len(car_nothing_train)} negative")
    print(f"   👤 Human: {len(human_train)} positive, {len(human_nothing_train)} negative")
    
    print(f"\n🧪 Test set sizes:")
    print(f"   🚗 Car: {len(car_test)} positive, {len(car_nothing_test)} negative")
    print(f"   👤 Human: {len(human_test)} positive, {len(human_nothing_test)} negative")
    
    # Train classifiers
    print(f"\n" + "="*50)
    print("🎓 TRAINING PHASE")
    print("="*50)
    
    car_train_acc = classifier.train_binary_classifier(
        positive_spikegrams=car_train,
        negative_spikegrams=car_nothing_train,
        signal_type='car'
    )
    
    human_train_acc = classifier.train_binary_classifier(
        positive_spikegrams=human_train,
        negative_spikegrams=human_nothing_train,
        signal_type='human'
    )
    
    # Test classifiers
    print(f"\n" + "="*50)
    print("🧪 TESTING PHASE")
    print("="*50)
    
    car_results = evaluate_classifier(
        classifier=classifier,
        test_positive=car_test,
        test_negative=car_nothing_test,
        signal_type='car'
    )
    
    human_results = evaluate_classifier(
        classifier=classifier,
        test_positive=human_test,
        test_negative=human_nothing_test,
        signal_type='human'
    )
    
    # Final results summary
    print(f"\n" + "="*70)
    print("🎉 FINAL RESULTS")
    print("="*70)
    
    print(f"\n🚗 CAR CLASSIFIER:")
    print(f"   Training Accuracy: {car_train_acc:.1%}")
    print(f"   Test Accuracy:     {car_results['accuracy']:.1%}")
    print(f"   Precision:         {car_results['precision']:.1%}")
    print(f"   Recall:            {car_results['recall']:.1%}")
    print(f"   F1-Score:          {car_results['f1']:.1%}")
    
    print(f"\n👤 HUMAN CLASSIFIER:")
    print(f"   Training Accuracy: {human_train_acc:.1%}")
    print(f"   Test Accuracy:     {human_results['accuracy']:.1%}")
    print(f"   Precision:         {human_results['precision']:.1%}")
    print(f"   Recall:            {human_results['recall']:.1%}")
    print(f"   F1-Score:          {human_results['f1']:.1%}")
    
    print(f"\n📈 SUMMARY:")
    if car_results['accuracy'] > 0.8 and human_results['accuracy'] > 0.8:
        print("   ✅ EXCELLENT: Both classifiers performing well!")
    elif car_results['accuracy'] > 0.7 or human_results['accuracy'] > 0.7:
        print("   👍 GOOD: At least one classifier performing well")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: Both classifiers need tuning")
    
    # Show confidence distribution
    print(f"\n📊 Confidence Analysis:")
    car_pos_conf = np.mean([c for c, l in zip(car_results['confidences'], 
                                             [True]*len(car_test) + [False]*len(car_nothing_test)) if l])
    car_neg_conf = np.mean([c for c, l in zip(car_results['confidences'], 
                                             [True]*len(car_test) + [False]*len(car_nothing_test)) if not l])
    
    print(f"   🚗 Car: Positive avg confidence = {car_pos_conf:.3f}, Negative avg confidence = {car_neg_conf:.3f}")
    
    human_pos_conf = np.mean([c for c, l in zip(human_results['confidences'], 
                                               [True]*len(human_test) + [False]*len(human_nothing_test)) if l])
    human_neg_conf = np.mean([c for c, l in zip(human_results['confidences'], 
                                               [True]*len(human_test) + [False]*len(human_nothing_test)) if not l])
    
    print(f"   👤 Human: Positive avg confidence = {human_pos_conf:.3f}, Negative avg confidence = {human_neg_conf:.3f}")
    
    return classifier, car_results, human_results

if __name__ == "__main__":
    try:
        classifier, car_results, human_results = run_footprint_classification()
        print("\n✅ Classification completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc() 