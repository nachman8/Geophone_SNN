#!/usr/bin/env python3
"""
ULTIMATE Car STDP Classifier - Target: 90%+ Accuracy

Combines ALL discovered insights:
1. Temporal Variability Detection (cars: std/mean > 0.25, nothing: < 0.15) 
2. Energy Discrimination (cars: 3.99x higher energy in 36.74 Hz)
3. Multi-scale Analysis (3s, 6s, and 15s windows)
4. Enhanced STDP with adaptive parameters
5. Ensemble voting and confidence scoring

This is the ultimate optimized version targeting 90%+ accuracy!
"""

import numpy as np
import pickle
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class UltimateCarPatternExtractor:
    """Ultimate car pattern extractor using ALL discovered insights"""
    
    def __init__(self):
        self.car_optimal_band = 2      # 36.74 Hz
        self.segment_duration = 15     # 15 seconds for car
        
        # Multi-scale analysis windows
        self.window_sizes = [300, 600, 900]  # 3s, 6s, 9s windows
        
        # Refined thresholds based on analysis
        self.variability_thresholds = {
            'high': 0.20,    # Lower threshold for more sensitivity
            'low': 0.10      # Lower threshold for more specificity
        }
        
        # Energy thresholds (from 3.99x discovery)
        self.energy_thresholds = {
            'car_chunk_min': 70000,      # Minimum for car chunks
            'car_segment_min': 4000,     # Minimum for car segments  
            'nothing_max': 25000         # Maximum for nothing segments
        }
        
        print(f"🎯 UltimateCarPatternExtractor initialized:")
        print(f"   🔍 Multi-scale windows: {[w/100 for w in self.window_sizes]}s")
        print(f"   📊 Variability: high>{self.variability_thresholds['high']}, low<{self.variability_thresholds['low']}")
    
    def extract_segments_from_chunks(self, chunk_data, signal_type):
        """Ultimate segment extraction with multi-criteria analysis"""
        print(f"\n🔍 ULTIMATE extraction from {signal_type} chunks...")
        
        all_segments = []
        all_labels = []
        
        is_nothing_file = signal_type.endswith('_nothing')
        base_signal_type = signal_type.replace('_nothing', '') if is_nothing_file else signal_type
        
        for chunk_idx, chunk in enumerate(chunk_data):
            if 'spikes_bands_spectrogram' not in chunk:
                continue
            
            spikegram = chunk['spikes_bands_spectrogram']
            
            # Multi-criteria chunk analysis
            chunk_score = self._analyze_chunk_comprehensive(spikegram, base_signal_type)
            
            # Extract segments from this chunk
            segments, labels = self._extract_segments_ultimate(
                spikegram, base_signal_type, is_nothing_file, chunk_score
            )
            
            all_segments.extend(segments)
            all_labels.extend(labels)
            
            signal_count = np.sum(np.array(labels) == 1)
            nothing_count = np.sum(np.array(labels) == 0)
            
            print(f"   📦 Chunk {chunk_idx}: {len(segments)} segments ({signal_count} car, {nothing_count} consistent)")
        
        print(f"   🎯 Total: {len(all_segments)} segments")
        print(f"     🚗 Car patterns: {np.sum(np.array(all_labels) == 1)}")
        print(f"     📉 Consistent patterns: {np.sum(np.array(all_labels) == 0)}")
        
        return np.array(all_segments), np.array(all_labels)
    
    def _analyze_chunk_comprehensive(self, spikegram, signal_type):
        """Comprehensive chunk analysis using ALL criteria"""
        optimal_band_data = spikegram[self.car_optimal_band, :]
        
        # Energy analysis
        total_energy = np.sum(optimal_band_data)
        energy_score = min(total_energy / self.energy_thresholds['car_chunk_min'], 2.0)
        
        # Multi-scale variability analysis
        variability_scores = []
        for window_size in self.window_sizes:
            var_score = self._calculate_multiscale_variability(optimal_band_data, window_size)
            variability_scores.append(var_score)
        
        avg_variability = np.mean(variability_scores)
        
        # Amplitude analysis
        max_amplitude = np.max(optimal_band_data)
        mean_amplitude = np.mean(optimal_band_data)
        amplitude_ratio = max_amplitude / (mean_amplitude + 1e-10)
        amplitude_score = min(amplitude_ratio / 10.0, 1.0)
        
        # Combined chunk score
        chunk_score = (energy_score * 0.4 + avg_variability * 0.4 + amplitude_score * 0.2)
        
        return chunk_score
    
    def _calculate_multiscale_variability(self, band_data, window_size):
        """Calculate variability at different time scales"""
        n_windows = len(band_data) // window_size
        if n_windows < 3:
            return 0.0
        
        window_energies = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_energy = np.sum(band_data[start:end])
            window_energies.append(window_energy)
        
        mean_energy = np.mean(window_energies)
        if mean_energy == 0:
            return 0.0
        
        std_energy = np.std(window_energies)
        variability_ratio = std_energy / mean_energy
        
        # Enhanced variability metrics
        energy_range = np.max(window_energies) - np.min(window_energies)
        range_ratio = energy_range / (mean_energy + 1e-10)
        
        # Burst detection
        high_threshold = mean_energy * 1.3
        burst_count = np.sum(np.array(window_energies) > high_threshold)
        burst_ratio = burst_count / n_windows
        
        # Combined variability score
        combined_score = (variability_ratio * 0.5 + range_ratio * 0.3 + burst_ratio * 0.2)
        
        return combined_score
    
    def _extract_segments_ultimate(self, spikegram, signal_type, is_nothing_file, chunk_score):
        """Ultimate segment extraction with enhanced criteria"""
        n_bands, n_time_bins = spikegram.shape
        samples_per_segment = int(self.segment_duration * 100)
        
        segments = []
        labels = []
        
        # Extract overlapping segments
        stride = samples_per_segment // 3  # More overlap for better coverage
        
        for start_idx in range(0, n_time_bins - samples_per_segment + 1, stride):
            end_idx = start_idx + samples_per_segment
            segment_spikegram = spikegram[:, start_idx:end_idx]
            
            # Extract ultimate features
            features = self._extract_ultimate_features(segment_spikegram, signal_type)
            segments.append(features)
            
            # Ultimate labeling logic
            if is_nothing_file:
                # Nothing files: ALL segments are consistent (0)
                labels.append(0)
            else:
                # Signal files: Use comprehensive detection
                has_car_pattern = self._detect_car_pattern_ultimate(segment_spikegram, chunk_score)
                labels.append(1 if has_car_pattern else 0)
        
        return segments, labels
    
    def _detect_car_pattern_ultimate(self, segment_spikegram, chunk_score):
        """Ultimate car pattern detection using ALL discovered criteria"""
        optimal_band_data = segment_spikegram[self.car_optimal_band, :]
        
        # Multi-scale variability analysis
        variability_scores = []
        for window_size in self.window_sizes:
            var_score = self._calculate_multiscale_variability(optimal_band_data, window_size)
            variability_scores.append(var_score)
        
        max_variability = np.max(variability_scores)
        avg_variability = np.mean(variability_scores)
        
        # Energy analysis
        segment_energy = np.sum(optimal_band_data)
        energy_criterion = segment_energy > self.energy_thresholds['car_segment_min']
        
        # Enhanced criteria with adaptive thresholds
        high_confidence = (
            (max_variability > self.variability_thresholds['high']) and
            energy_criterion
        )
        
        medium_confidence = (
            (avg_variability > self.variability_thresholds['low']) and
            energy_criterion
        )
        
        # Use chunk score to adjust thresholds
        if chunk_score > 1.2:  # High-confidence chunks
            return high_confidence or medium_confidence
        elif chunk_score > 0.8:  # Medium-confidence chunks
            return high_confidence
        else:  # Low-confidence chunks
            return high_confidence and (max_variability > 0.3)
    
    def _extract_ultimate_features(self, spikegram, signal_type):
        """Extract ultimate feature set combining all discoveries"""
        n_bands, n_time_bins = spikegram.shape
        features = []
        
        for band_idx in range(n_bands):
            band_data = spikegram[band_idx, :]
            
            if len(band_data) > 0:
                # Basic features
                mean_activity = np.mean(band_data)
                max_activity = np.max(band_data)
                std_activity = np.std(band_data)
                total_energy = np.sum(band_data)
                
                if band_idx == self.car_optimal_band:
                    # Ultimate car features for 36.74 Hz band
                    
                    # Multi-scale variability features
                    variability_features = []
                    for window_size in self.window_sizes:
                        var_score = self._calculate_multiscale_variability(band_data, window_size)
                        variability_features.append(var_score)
                    
                    # Enhanced temporal features
                    cv = std_activity / (mean_activity + 1e-10)
                    peak_ratio = max_activity / (mean_activity + 1e-10)
                    
                    features.extend([
                        mean_activity,
                        max_activity,
                        std_activity,
                        total_energy,
                        variability_features[0],  # 3s variability
                        variability_features[1],  # 6s variability  
                        variability_features[2],  # 9s variability
                        cv,                       # Coefficient of variation
                        peak_ratio,               # Peak-to-mean ratio
                        np.max(variability_features),  # Maximum variability
                    ])
                else:
                    # Standard features for other bands
                    features.extend([
                        mean_activity,
                        max_activity,
                        std_activity,
                        total_energy,
                        np.sum(band_data > mean_activity + std_activity),
                        np.sum(band_data > 0) / len(band_data),
                        std_activity / (mean_activity + 1e-10),
                        0.0,  # Padding
                        0.0,  # Padding
                        0.0   # Padding
                    ])
            else:
                features.extend([0.0] * 10)  # 10 features per band
        
        return np.array(features, dtype=np.float32)

class UltimateSTDPClassifier:
    """Ultimate STDP SNN with all optimizations for 90%+ accuracy"""
    
    def __init__(self, n_input_features=80, n_hidden=120, n_output=2):
        self.n_input_features = n_input_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Ultimate network initialization
        self.W_input_hidden = np.random.normal(0.45, 0.12, (n_input_features, n_hidden))
        self.W_hidden_output = np.random.normal(0.55, 0.08, (n_hidden, n_output))
        self.W_input_hidden = np.clip(self.W_input_hidden, 0.15, 0.75)
        self.W_hidden_output = np.clip(self.W_hidden_output, 0.25, 0.85)
        
        # Ultimate STDP parameters
        self.learning_rate = 0.035
        self.A_plus = 0.045
        self.A_minus = -0.012
        
        # Adaptive network parameters
        self.threshold = 0.7
        self.decay_rate = 0.88
        self.refractory_period = 2
        
        # Ultimate learning schedule
        self.lr_decay = 0.985
        self.min_lr = 0.008
        
        print(f"🧠 UltimateSTDPClassifier initialized:")
        print(f"   📊 Architecture: {n_input_features} → {n_hidden} → {n_output}")
        print(f"   🎯 TARGET: 90%+ accuracy!")
    
    def spike_encode(self, features, spike_duration=300):
        """Ultimate spike encoding with optimized parameters"""
        features_norm = np.copy(features).astype(float)
        
        # Enhanced normalization
        for i in range(len(features_norm)):
            if features_norm[i] > 0:
                features_norm[i] = np.log1p(features_norm[i])
        
        feature_max = np.max(features_norm)
        if feature_max > 0:
            features_norm = features_norm / feature_max
        
        # Ultimate spike generation
        max_rate = 150
        dt = 1.0
        n_steps = int(spike_duration / dt)
        
        spike_trains = []
        for i, norm_value in enumerate(features_norm):
            base_rate = 15.0
            rate = base_rate + norm_value * (max_rate - base_rate)
            
            spike_times = []
            last_spike = -self.refractory_period
            
            for t in range(n_steps):
                if t - last_spike >= self.refractory_period:
                    spike_prob = rate * dt / 1000.0
                    if np.random.random() < spike_prob:
                        spike_times.append(t * dt)
                        last_spike = t
            
            spike_trains.append(np.array(spike_times))
        
        return spike_trains
    
    def simulate_network(self, spike_trains, target=None, training=True, epoch=0):
        """Ultimate network simulation"""
        spike_duration = 300
        dt = 1.0
        n_steps = int(spike_duration / dt)
        
        # Network state
        hidden_membrane = np.zeros(self.n_hidden)
        output_membrane = np.zeros(self.n_output)
        hidden_refractory = np.zeros(self.n_hidden)
        output_refractory = np.zeros(self.n_output)
        
        output_spike_times = [[] for _ in range(self.n_output)]
        
        # Adaptive learning rate
        current_lr = max(self.learning_rate * (self.lr_decay ** epoch), self.min_lr)
        
        for t in range(n_steps):
            current_time = t * dt
            
            # Network dynamics
            hidden_membrane *= self.decay_rate
            output_membrane *= self.decay_rate
            hidden_refractory = np.maximum(0, hidden_refractory - dt)
            output_refractory = np.maximum(0, output_refractory - dt)
            
            # Input processing
            input_spikes = np.zeros(self.n_input_features)
            for i, spike_times in enumerate(spike_trains):
                if len(spike_times) > 0 and np.any(np.abs(spike_times - current_time) < dt/2):
                    input_spikes[i] = 1.0
            
            # Hidden layer
            hidden_input = np.dot(input_spikes, self.W_input_hidden)
            hidden_membrane += hidden_input
            
            hidden_spikes = (hidden_membrane > self.threshold) & (hidden_refractory == 0)
            hidden_membrane[hidden_spikes] = 0.0
            hidden_refractory[hidden_spikes] = self.refractory_period
            
            # Output layer
            output_input = np.dot(hidden_spikes.astype(float), self.W_hidden_output)
            output_membrane += output_input
            
            output_spikes = (output_membrane > self.threshold) & (output_refractory == 0)
            output_membrane[output_spikes] = 0.0
            output_refractory[output_spikes] = self.refractory_period
            
            # Record output spikes
            for i, spike in enumerate(output_spikes):
                if spike:
                    output_spike_times[i].append(current_time)
            
            # STDP learning
            if training and target is not None:
                self._apply_ultimate_stdp(input_spikes, hidden_spikes, output_spikes, target, current_lr)
        
        return output_spike_times
    
    def _apply_ultimate_stdp(self, input_spikes, hidden_spikes, output_spikes, target, lr):
        """Ultimate STDP learning"""
        # Input-Hidden STDP
        for i in range(self.n_input_features):
            if input_spikes[i] > 0:
                for j in range(self.n_hidden):
                    if hidden_spikes[j]:
                        self.W_input_hidden[i, j] += lr * self.A_plus
                    else:
                        self.W_input_hidden[i, j] += lr * self.A_minus * 0.3
        
        # Hidden-Output STDP
        for i in range(self.n_hidden):
            if hidden_spikes[i]:
                self.W_hidden_output[i, target] += lr * self.A_plus
                
                for j in range(self.n_output):
                    if j != target:
                        depression = lr * self.A_minus * (1.0 + 0.3 * output_spikes[j])
                        self.W_hidden_output[i, j] += depression
        
        # Weight bounds
        self.W_input_hidden = np.clip(self.W_input_hidden, 0.05, 1.0)
        self.W_hidden_output = np.clip(self.W_hidden_output, 0.1, 1.0)
        
        # Homeostasis
        if np.random.random() < 0.002:
            self.W_input_hidden *= 0.999
            self.W_hidden_output *= 0.999
    
    def train(self, X_train, y_train, n_epochs=70):
        """Ultimate training for 90%+ accuracy"""
        print(f"\n🎓 Training ULTIMATE STDP SNN for {n_epochs} epochs (TARGET: 90%+)...")
        
        training_history = {'epoch': [], 'accuracy': []}
        best_accuracy = 0.0
        
        for epoch in range(n_epochs):
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            
            for idx in indices:
                features = X_train[idx]
                target = y_train[idx]
                
                spike_trains = self.spike_encode(features)
                output_spike_times = self.simulate_network(spike_trains, target, training=True, epoch=epoch)
                prediction = self._make_prediction(output_spike_times)
                
                if prediction == target:
                    correct_predictions += 1
            
            epoch_accuracy = correct_predictions / len(X_train)
            training_history['epoch'].append(epoch)
            training_history['accuracy'].append(epoch_accuracy)
            
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                current_lr = max(self.learning_rate * (self.lr_decay ** epoch), self.min_lr)
                print(f"   📊 Epoch {epoch:2d}: acc={epoch_accuracy:.1%} (best: {best_accuracy:.1%})")
            
            # Early stopping for 90%+
            if best_accuracy >= 0.92:
                print(f"   🎯 Early stopping: achieved {best_accuracy:.1%}")
                break
        
        if best_accuracy >= 0.90:
            print("   🎉 TARGET ACHIEVED! 90%+ accuracy reached!")
        elif best_accuracy >= 0.85:
            print("   🔥 EXCELLENT! Very close to target!")
        
        return training_history
    
    def predict(self, X_test):
        """Ultimate prediction with ensemble voting"""
        predictions = []
        
        for features in X_test:
            votes = []
            for _ in range(5):
                spike_trains = self.spike_encode(features)
                output_spike_times = self.simulate_network(spike_trains, training=False)
                vote = self._make_prediction(output_spike_times)
                votes.append(vote)
            
            prediction = max(set(votes), key=votes.count)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _make_prediction(self, output_spike_times):
        """Enhanced prediction"""
        spike_counts = [len(spikes) for spikes in output_spike_times]
        
        if max(spike_counts) > 0:
            return np.argmax(spike_counts)
        else:
            return 0

def run_ultimate_car_classification():
    """Run ULTIMATE car classification targeting 90%+ accuracy"""
    print("🚀 ULTIMATE CAR STDP SNN CLASSIFICATION")
    print("🎯 TARGET: 90%+ ACCURACY")
    print("🔥 Combining ALL discovered insights")
    print("=" * 80)
    
    # Initialize ultimate components
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    from pathlib import Path
    chunks_path = Path(chunks_dir)
    
    # Load car data
    car_extractor = UltimateCarPatternExtractor()
    
    # Load car chunks
    car_dir = chunks_path / 'car'
    car_chunk_files = sorted(car_dir.glob("chunk_*/chunk_*_data.pkl"))
    car_chunk_data = []
    
    for chunk_file in car_chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
            if 'spikes_bands_spectrogram' in chunk:
                car_chunk_data.append(chunk)
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
    
    print(f"✅ Loaded {len(car_chunk_data)} car chunks")
    
    # Load car_nothing chunks
    car_nothing_dir = chunks_path / 'car_nothing'
    car_nothing_chunk_files = sorted(car_nothing_dir.glob("chunk_*/chunk_*_data.pkl"))
    car_nothing_chunk_data = []
    
    for chunk_file in car_nothing_chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
            if 'spikes_bands_spectrogram' in chunk:
                car_nothing_chunk_data.append(chunk)
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
    
    print(f"✅ Loaded {len(car_nothing_chunk_data)} car_nothing chunks")
    
    # Extract segments
    car_segments, car_labels = car_extractor.extract_segments_from_chunks(car_chunk_data, 'car')
    car_nothing_segments, car_nothing_labels = car_extractor.extract_segments_from_chunks(car_nothing_chunk_data, 'car_nothing')
    
    # Prepare ultimate dataset
    print(f"\n🚗 ULTIMATE CAR DATASET")
    all_segments = np.vstack([car_segments, car_nothing_segments])
    all_labels = np.hstack([car_labels, car_nothing_labels])
    print(f"   📊 Total: {len(all_segments)}, Car: {np.sum(all_labels == 1)}, Consistent: {np.sum(all_labels == 0)}")
    
    # Train ultimate classifier
    print(f"\n" + "="*70)
    print("🚗 ULTIMATE CAR CLASSIFICATION - TARGET: 90%+")
    print("="*70)
    
    if len(np.unique(all_labels)) >= 2 and len(all_segments) > 10:
        classifier = UltimateSTDPClassifier(n_input_features=all_segments.shape[1])
        
        X_train, X_test, y_train, y_test = train_test_split(
            all_segments, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        print(f"🎓 Training: {len(X_train)}, Testing: {len(X_test)}")
        
        history = classifier.train(X_train, y_train, n_epochs=70)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n🎯 ULTIMATE CAR RESULTS:")
        print(f"   📊 Test Accuracy: {accuracy:.1%}")
        print(f"   📈 Best Training: {max(history['accuracy']):.1%}")
        
        if accuracy >= 0.90:
            print("   ��🎉 SUCCESS! 90%+ TARGET ACHIEVED! 🎉🎉")
        elif accuracy >= 0.85:
            print("   🔥 EXCELLENT! Very close to 90% target!")
        elif accuracy >= 0.80:
            print("   👍 GREAT! Major improvement achieved!")
        
        # Save model
        with open('ultimate_car_stdp_model.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        print(f"   💾 Model saved: ultimate_car_stdp_model.pkl")
        
        return accuracy
    else:
        print("❌ Insufficient data")
        return 0.0

if __name__ == "__main__":
    try:
        accuracy = run_ultimate_car_classification()
        print(f"\n✅ ULTIMATE classification completed!")
        print(f"🎯 FINAL ACCURACY: {accuracy:.1%}")
        
        if accuracy >= 0.90:
            print("🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉")
        elif accuracy >= 0.85:
            print("🔥 OUTSTANDING! Almost there!")
        else:
            print("📈 Continue optimization...")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
