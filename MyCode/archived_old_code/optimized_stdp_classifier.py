#!/usr/bin/env python3
"""
Optimized STDP SNN Classifier Based on Discovered Footprint Patterns

Key Discoveries:
🚗 Car: 3.99x energy ratio in 36.74 Hz band (strongest discriminator)
�� Human: Organized footsteps (40-51 per 120s) with 0.5-3s intervals
📊 Strong amplitude differences and organized temporal patterns

Target: 90%+ accuracy through:
1. Energy-based discrimination (primary feature)
2. Organized pattern detection (footstep timing)
3. Enhanced STDP learning with better features
4. Optimal frequency focus (36.74 Hz car, 66.16 Hz human)
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

class OptimizedFootprintExtractor:
    """Extract footprint patterns based on discovered optimal features"""
    
    def __init__(self):
        # Optimal frequency bands (from analysis)
        self.car_optimal_band = 2      # 36.74 Hz
        self.human_optimal_band = 5    # 66.16 Hz
        
        # Discovered thresholds (from actual data analysis)
        self.car_energy_threshold = 120000      # Car chunks have ~164k energy, nothing ~41k
        self.human_energy_threshold = 40000     # Human chunks have ~45-62k energy
        
        # Footstep pattern parameters
        self.min_footsteps_per_chunk = 25       # Minimum organized footsteps
        self.max_step_interval = 5.0            # Max 5 seconds between steps
        self.min_step_interval = 0.3            # Min 0.3 seconds between steps
        
        # Segment parameters  
        self.car_segment_duration = 15
        self.human_segment_duration = 7
        
        print(f"🎯 OptimizedFootprintExtractor initialized with discovered patterns:")
        print(f"   🚗 Car energy threshold: {self.car_energy_threshold:,} (36.74 Hz)")
        print(f"   👤 Human energy threshold: {self.human_energy_threshold:,} (66.16 Hz)")
        print(f"   👣 Footstep detection: {self.min_footsteps_per_chunk}-51 steps per chunk")
    
    def extract_segments_from_chunks(self, chunk_data, signal_type):
        """Extract segments with OPTIMIZED footprint detection"""
        print(f"\n🔍 Extracting segments from {signal_type} chunks with optimized detection...")
        
        all_segments = []
        all_labels = []
        
        is_nothing_file = signal_type.endswith('_nothing')
        base_signal_type = signal_type.replace('_nothing', '') if is_nothing_file else signal_type
        segment_duration = self.car_segment_duration if base_signal_type == 'car' else self.human_segment_duration
        
        print(f"   📋 File type: {'NOTHING (background only)' if is_nothing_file else 'SIGNAL (contains patterns)'}")
        
        for chunk_idx, chunk in enumerate(chunk_data):
            if 'spikes_bands_spectrogram' not in chunk:
                continue
            
            spikegram = chunk['spikes_bands_spectrogram']
            print(f"   📦 Chunk {chunk_idx}: {spikegram.shape}")
            
            # Analyze entire chunk first for global pattern detection
            chunk_footprint_score = self._analyze_entire_chunk(spikegram, base_signal_type)
            
            # Extract segments from this chunk
            segments, labels = self._extract_segments_from_spikegram(
                spikegram, base_signal_type, is_nothing_file, segment_duration, chunk_footprint_score
            )
            
            all_segments.extend(segments)
            all_labels.extend(labels)
            
            signal_count = np.sum(np.array(labels) == 1)
            nothing_count = np.sum(np.array(labels) == 0)
            
            print(f"     ✅ {len(segments)} segments: {signal_count} patterns, {nothing_count} background")
            print(f"     📊 Chunk footprint score: {chunk_footprint_score:.3f}")
        
        print(f"   🎯 Total: {len(all_segments)} segments")
        print(f"     📈 Pattern segments: {np.sum(np.array(all_labels) == 1)}")
        print(f"     📉 Background segments: {np.sum(np.array(all_labels) == 0)}")
        
        return np.array(all_segments), np.array(all_labels)
    
    def _analyze_entire_chunk(self, spikegram, signal_type):
        """Analyze entire chunk to determine overall footprint strength"""
        if signal_type == 'car':
            # Car: Check energy in 36.74 Hz band
            optimal_band_data = spikegram[self.car_optimal_band, :]
            total_energy = np.sum(optimal_band_data)
            
            # Energy-based scoring (car has 3.99x more energy than car_nothing)
            energy_score = min(total_energy / self.car_energy_threshold, 2.0)
            
            # Amplitude-based scoring
            max_amplitude = np.max(optimal_band_data)
            amplitude_score = min(max_amplitude / 100.0, 2.0)  # Car max ~128
            
            footprint_score = (energy_score + amplitude_score) / 2.0
            
        elif signal_type == 'human':
            # Human: Check organized footstep patterns in 66.16 Hz band
            optimal_band_data = spikegram[self.human_optimal_band, :]
            
            # Energy-based scoring
            total_energy = np.sum(optimal_band_data)
            energy_score = min(total_energy / self.human_energy_threshold, 2.0)
            
            # Footstep organization scoring
            footstep_score = self._detect_organized_footsteps(optimal_band_data)
            
            # Combined scoring
            footprint_score = (energy_score + footstep_score) / 2.0
            
        else:
            footprint_score = 0.0
        
        return footprint_score
    
    def _detect_organized_footsteps(self, band_data):
        """Detect organized footstep patterns (key insight from analysis)"""
        mean_val = np.mean(band_data)
        std_val = np.std(band_data)
        
        # Use 4-sigma threshold for strong footstep impacts (from analysis)
        strong_threshold = mean_val + 4 * std_val
        strong_peaks = band_data > strong_threshold
        strong_peak_indices = np.where(strong_peaks)[0]
        
        if len(strong_peak_indices) < 5:
            return 0.0  # Too few peaks for organized walking
        
        # Group nearby peaks as single footsteps (within 0.5 seconds)
        footsteps = []
        current_step_start = strong_peak_indices[0]
        
        for i in range(1, len(strong_peak_indices)):
            if strong_peak_indices[i] - strong_peak_indices[i-1] > 50:  # 0.5 second gap
                footsteps.append(current_step_start)
                current_step_start = strong_peak_indices[i]
        footsteps.append(current_step_start)
        
        num_footsteps = len(footsteps)
        
        # Check if footstep count is in realistic range (25-60 steps per 120s chunk)
        if num_footsteps < self.min_footsteps_per_chunk:
            return 0.0
        
        # Check step intervals for realistic walking pattern
        if len(footsteps) > 1:
            step_intervals = np.diff(np.array(footsteps)) / 100.0  # Convert to seconds
            
            # Filter realistic intervals (0.3-5 seconds)
            valid_intervals = step_intervals[
                (step_intervals >= self.min_step_interval) & 
                (step_intervals <= self.max_step_interval)
            ]
            
            if len(valid_intervals) < len(step_intervals) * 0.7:
                return 0.0  # Too many unrealistic intervals
            
            # Score based on footstep count and interval consistency
            count_score = min(num_footsteps / 45.0, 1.0)  # Target ~45 steps
            consistency_score = 1.0 / (1.0 + np.std(valid_intervals))
            
            return (count_score + consistency_score) / 2.0
        
        return 0.0
    
    def _extract_segments_from_spikegram(self, spikegram, signal_type, is_nothing_file, segment_duration, chunk_score):
        """Extract segments with optimized labeling based on discovered patterns"""
        n_bands, n_time_bins = spikegram.shape
        samples_per_segment = int(segment_duration * 100)
        
        segments = []
        labels = []
        
        # Extract overlapping segments
        stride = samples_per_segment // 2
        
        for start_idx in range(0, n_time_bins - samples_per_segment + 1, stride):
            end_idx = start_idx + samples_per_segment
            segment_spikegram = spikegram[:, start_idx:end_idx]
            
            # Convert to enhanced features
            features = self._extract_optimized_features(segment_spikegram, signal_type)
            segments.append(features)
            
            # OPTIMIZED labeling logic
            if is_nothing_file:
                # Nothing files: ALL segments are background (0)
                labels.append(0)
            else:
                # Signal files: Use discovered patterns for detection
                segment_score = self._score_segment_footprint(segment_spikegram, signal_type, chunk_score)
                
                # Use dynamic threshold based on signal type
                if signal_type == 'car':
                    threshold = 0.6  # Car patterns are stronger (3.99x energy difference)
                elif signal_type == 'human':
                    threshold = 0.4  # Human patterns need lower threshold (1.63x energy difference)
                else:
                    threshold = 0.5
                
                has_pattern = segment_score > threshold
                labels.append(1 if has_pattern else 0)
        
        return segments, labels
    
    def _score_segment_footprint(self, segment_spikegram, signal_type, chunk_score):
        """Score individual segment using discovered optimal features"""
        if signal_type == 'car':
            # Car: Focus on energy and amplitude in 36.74 Hz band
            optimal_band_data = segment_spikegram[self.car_optimal_band, :]
            
            # Energy in optimal band
            energy = np.sum(optimal_band_data)
            energy_score = min(energy / (self.car_energy_threshold * 0.125), 1.0)  # 1/8 of chunk
            
            # Max amplitude
            max_amp = np.max(optimal_band_data)
            amp_score = min(max_amp / 100.0, 1.0)
            
            # Mean activity
            mean_activity = np.mean(optimal_band_data)
            mean_score = min(mean_activity / 10.0, 1.0)
            
            segment_score = (energy_score * 0.5 + amp_score * 0.3 + mean_score * 0.2) * chunk_score
            
        elif signal_type == 'human':
            # Human: Focus on footstep impacts in 66.16 Hz band
            optimal_band_data = segment_spikegram[self.human_optimal_band, :]
            
            # Energy in optimal band
            energy = np.sum(optimal_band_data)
            energy_score = min(energy / (self.human_energy_threshold * 0.058), 1.0)  # 1/17 of chunk
            
            # Strong impact detection
            mean_val = np.mean(optimal_band_data)
            std_val = np.std(optimal_band_data)
            strong_threshold = mean_val + 3 * std_val  # Lower threshold for segments
            strong_impacts = np.sum(optimal_band_data > strong_threshold)
            impact_score = min(strong_impacts / 10.0, 1.0)
            
            # Max amplitude
            max_amp = np.max(optimal_band_data)
            amp_score = min(max_amp / 50.0, 1.0)
            
            segment_score = (energy_score * 0.4 + impact_score * 0.4 + amp_score * 0.2) * chunk_score
            
        else:
            segment_score = 0.0
        
        return segment_score
    
    def _extract_optimized_features(self, spikegram, signal_type):
        """Extract optimized features based on discovered patterns"""
        n_bands, n_time_bins = spikegram.shape
        features = []
        
        # Extract features for each frequency band
        for band_idx in range(n_bands):
            band_data = spikegram[band_idx, :]
            
            if len(band_data) > 0:
                # Basic features
                mean_activity = np.mean(band_data)
                max_activity = np.max(band_data)
                std_activity = np.std(band_data)
                total_energy = np.sum(band_data)
                
                # Enhanced features based on discoveries
                if band_idx == self.car_optimal_band and signal_type == 'car':
                    # Car-specific features for 36.74 Hz band
                    features.extend([
                        mean_activity,
                        max_activity,
                        std_activity,
                        total_energy,
                        total_energy / (mean_activity + 1e-10),  # Energy concentration
                        max_activity / (mean_activity + 1e-10),  # Peak ratio
                        np.percentile(band_data, 90),
                        np.sum(band_data > mean_activity + 2*std_activity),  # Strong activity count
                        mean_activity / 13.64,  # Normalized by car mean (from analysis)
                        max_activity / 128.17   # Normalized by car max (from analysis)
                    ])
                elif band_idx == self.human_optimal_band and signal_type == 'human':
                    # Human-specific features for 66.16 Hz band
                    strong_threshold = mean_activity + 3 * std_activity
                    strong_impacts = np.sum(band_data > strong_threshold)
                    
                    features.extend([
                        mean_activity,
                        max_activity,
                        std_activity,
                        total_energy,
                        strong_impacts,  # Footstep impact count
                        strong_impacts / len(band_data),  # Impact density
                        np.percentile(band_data, 95),  # High percentile for impacts
                        max_activity / (mean_activity + 1e-10),  # Impact strength ratio
                        mean_activity / 4.5,   # Normalized by human mean (from analysis)
                        max_activity / 65.0    # Normalized by human max (from analysis)
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
                        np.percentile(band_data, 90),
                        np.sum(np.diff(band_data) > 0) / len(band_data) if len(band_data) > 1 else 0,
                        0.0,  # Placeholder
                        0.0   # Placeholder
                    ])
            else:
                features.extend([0.0] * 10)  # 10 features per band
        
        return np.array(features, dtype=np.float32)

class EnhancedSTDPClassifier:
    """Enhanced STDP SNN with optimized architecture for 90%+ accuracy"""
    
    def __init__(self, n_input_features=80, n_hidden=120, n_output=2):
        self.n_input_features = n_input_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Optimized weight initialization for better learning
        self.W_input_hidden = np.random.normal(0.5, 0.1, (n_input_features, n_hidden))
        self.W_hidden_output = np.random.normal(0.5, 0.1, (n_hidden, n_output))
        self.W_input_hidden = np.clip(self.W_input_hidden, 0.2, 0.8)
        self.W_hidden_output = np.clip(self.W_hidden_output, 0.2, 0.8)
        
        # Enhanced STDP parameters for better convergence
        self.learning_rate = 0.02
        self.A_plus = 0.03          # Stronger LTP
        self.A_minus = -0.008       # Weaker LTD
        
        # Enhanced neuron parameters
        self.threshold = 1.0
        self.decay_rate = 0.9
        self.refractory_period = 5  # ms
        
        # Learning schedule
        self.lr_decay = 0.95
        self.min_lr = 0.005
        
        print(f"🧠 EnhancedSTDPClassifier initialized for 90%+ accuracy:")
        print(f"   📊 Architecture: {n_input_features} → {n_hidden} → {n_output}")
        print(f"   ⚡ Enhanced STDP: A+={self.A_plus}, A-={self.A_minus}")
        print(f"   🎯 Target: 90%+ test accuracy")
    
    def spike_encode(self, features, spike_duration=300):
        """Enhanced spike encoding with better temporal dynamics"""
        # Better normalization preserving relative magnitudes
        features_norm = np.copy(features)
        
        # Robust normalization
        for i in range(len(features_norm)):
            if features_norm[i] > 0:
                features_norm[i] = np.log1p(features_norm[i])  # Log transform for better distribution
        
        feature_max = np.max(features_norm)
        if feature_max > 0:
            features_norm = features_norm / feature_max
        
        # Enhanced spike generation
        max_rate = 150  # Higher max rate
        dt = 1.0
        n_steps = int(spike_duration / dt)
        
        spike_trains = []
        for i, norm_value in enumerate(features_norm):
            # Adaptive firing rate with minimum baseline
            base_rate = 8.0  # Higher baseline
            rate = base_rate + norm_value * (max_rate - base_rate)
            
            # Poisson spike generation with refractory period
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
        """Enhanced network simulation with adaptive learning"""
        spike_duration = 300
        dt = 1.0
        n_steps = int(spike_duration / dt)
        
        # Enhanced state tracking
        hidden_membrane = np.zeros(self.n_hidden)
        output_membrane = np.zeros(self.n_output)
        hidden_refractory = np.zeros(self.n_hidden)
        output_refractory = np.zeros(self.n_output)
        
        output_spike_times = [[] for _ in range(self.n_output)]
        
        # Adaptive learning rate
        current_lr = max(self.learning_rate * (self.lr_decay ** epoch), self.min_lr)
        
        for t in range(n_steps):
            current_time = t * dt
            
            # Decay and refractory updates
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
            
            # Apply threshold with refractory period
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
            
            # Enhanced STDP learning
            if training and target is not None:
                self._apply_enhanced_stdp(input_spikes, hidden_spikes, output_spikes, target, current_lr)
        
        return output_spike_times
    
    def _apply_enhanced_stdp(self, input_spikes, hidden_spikes, output_spikes, target, lr):
        """Enhanced STDP with competitive learning"""
        # Input-Hidden STDP
        for i in range(self.n_input_features):
            if input_spikes[i] > 0:
                for j in range(self.n_hidden):
                    if hidden_spikes[j]:
                        # Strengthen active connections
                        self.W_input_hidden[i, j] += lr * self.A_plus
                    else:
                        # Mild depression for inactive neurons
                        self.W_input_hidden[i, j] += lr * self.A_minus * 0.3
        
        # Hidden-Output STDP with winner-take-all
        total_hidden_activity = np.sum(hidden_spikes)
        if total_hidden_activity > 0:
            for i in range(self.n_hidden):
                if hidden_spikes[i]:
                    # Strengthen connection to target
                    self.W_hidden_output[i, target] += lr * self.A_plus
                    
                    # Competitive depression for non-target outputs
                    for j in range(self.n_output):
                        if j != target:
                            depression = lr * self.A_minus * (1.0 + 0.5 * output_spikes[j])
                            self.W_hidden_output[i, j] += depression
        
        # Weight normalization and clipping
        self.W_input_hidden = np.clip(self.W_input_hidden, 0.1, 1.0)
        self.W_hidden_output = np.clip(self.W_hidden_output, 0.1, 1.0)
        
        # Synaptic scaling for stability
        if np.random.random() < 0.01:  # 1% chance per timestep
            self.W_input_hidden *= 0.995
            self.W_hidden_output *= 0.995
    
    def train(self, X_train, y_train, n_epochs=50):
        """Enhanced training for 90%+ accuracy"""
        print(f"\n🎓 Training Enhanced STDP SNN for {n_epochs} epochs (target: 90%+)...")
        
        training_history = {'epoch': [], 'accuracy': [], 'lr': []}
        best_accuracy = 0.0
        patience = 15
        no_improve_count = 0
        
        for epoch in range(n_epochs):
            correct_predictions = 0
            current_lr = max(self.learning_rate * (self.lr_decay ** epoch), self.min_lr)
            
            # Shuffle and balance data
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
            training_history['lr'].append(current_lr)
            
            # Early stopping and best model tracking
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"   📊 Epoch {epoch:2d}: acc={epoch_accuracy:.1%} (best: {best_accuracy:.1%}) lr={current_lr:.4f}")
            
            # Early stopping for convergence
            if no_improve_count >= patience and epoch_accuracy > 0.85:
                print(f"   🎯 Early stopping: converged at {epoch_accuracy:.1%}")
                break
        
        final_message = "🎯 TARGET ACHIEVED!" if best_accuracy >= 0.90 else "📈 GOOD PROGRESS" if best_accuracy >= 0.80 else "🔧 NEEDS TUNING"
        print(f"   {final_message} Best accuracy: {best_accuracy:.1%}")
        
        return training_history
    
    def predict(self, X_test):
        """Enhanced prediction with ensemble voting"""
        predictions = []
        
        for features in X_test:
            # Multiple predictions for robustness
            votes = []
            for _ in range(3):  # 3 votes per sample
                spike_trains = self.spike_encode(features)
                output_spike_times = self.simulate_network(spike_trains, training=False)
                vote = self._make_prediction(output_spike_times)
                votes.append(vote)
            
            # Majority voting
            prediction = max(set(votes), key=votes.count)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _make_prediction(self, output_spike_times):
        """Enhanced prediction with confidence weighting"""
        spike_counts = [len(spikes) for spikes in output_spike_times]
        
        if max(spike_counts) > 0:
            return np.argmax(spike_counts)
        else:
            return 0

class OptimizedDataLoader:
    """Load and process data with optimized footprint extraction"""
    
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        self.extractor = OptimizedFootprintExtractor()
        
        print(f"📁 OptimizedDataLoader initialized for 90%+ accuracy")
    
    def load_and_extract_segments(self, signal_type):
        """Load chunks and extract optimized segments"""
        signal_dir = self.chunks_dir / signal_type
        
        if not signal_dir.exists():
            return None, None
        
        # Load chunk files
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
        
        # Extract with optimized methods
        segments, labels = self.extractor.extract_segments_from_chunks(chunk_data, signal_type)
        return segments, labels

def run_optimized_classification():
    """Run optimized classification targeting 90%+ accuracy"""
    print("🚀 OPTIMIZED STDP SNN Classification - Target: 90%+ Accuracy")
    print("🎯 Based on discovered footprint patterns:")
    print("   🚗 Car: 3.99x energy difference at 36.74 Hz")
    print("   👤 Human: Organized footsteps (40-51 per chunk) at 66.16 Hz")
    print("=" * 80)
    
    # Initialize optimized components
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    data_loader = OptimizedDataLoader(chunks_dir)
    
    # Load all data
    car_segments, car_labels = data_loader.load_and_extract_segments('car')
    car_nothing_segments, car_nothing_labels = data_loader.load_and_extract_segments('car_nothing')
    human_segments, human_labels = data_loader.load_and_extract_segments('human')
    human_nothing_segments, human_nothing_labels = data_loader.load_and_extract_segments('human_nothing')
    
    if car_segments is None or human_segments is None:
        print("❌ Data loading failed")
        return None
    
    # Prepare optimized datasets
    print(f"\n🚗 OPTIMIZED CAR DATASET")
    car_all_segments = np.vstack([car_segments, car_nothing_segments])
    car_all_labels = np.hstack([car_labels, car_nothing_labels])
    print(f"   📊 Total: {len(car_all_segments)}, Patterns: {np.sum(car_all_labels == 1)}, Background: {np.sum(car_all_labels == 0)}")
    
    print(f"\n👤 OPTIMIZED HUMAN DATASET")
    human_all_segments = np.vstack([human_segments, human_nothing_segments])
    human_all_labels = np.hstack([human_labels, human_nothing_labels])
    print(f"   📊 Total: {len(human_all_segments)}, Patterns: {np.sum(human_all_labels == 1)}, Background: {np.sum(human_all_labels == 0)}")
    
    results = {}
    
    # Train optimized car classifier
    print(f"\n" + "="*60)
    print("🚗 OPTIMIZED CAR CLASSIFICATION (Target: 90%+)")
    print("="*60)
    
    if len(np.unique(car_all_labels)) >= 2 and len(car_all_segments) > 10:
        car_classifier = EnhancedSTDPClassifier(n_input_features=car_all_segments.shape[1])
        
        X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
            car_all_segments, car_all_labels, test_size=0.25, random_state=42, stratify=car_all_labels
        )
        
        print(f"🎓 Training: {len(X_car_train)}, Testing: {len(X_car_test)}")
        
        car_history = car_classifier.train(X_car_train, y_car_train, n_epochs=60)
        car_predictions = car_classifier.predict(X_car_test)
        car_accuracy = accuracy_score(y_car_test, car_predictions)
        
        results['car'] = car_accuracy
        
        print(f"🎯 CAR RESULTS:")
        print(f"   📊 Test Accuracy: {car_accuracy:.1%}")
        print(f"   {'🎉 EXCELLENT!' if car_accuracy >= 0.90 else '👍 GOOD!' if car_accuracy >= 0.80 else '🔧 NEEDS IMPROVEMENT'}")
        
        with open('optimized_car_stdp_model.pkl', 'wb') as f:
            pickle.dump(car_classifier, f)
    else:
        print("❌ Insufficient car data")
        results['car'] = 0.0
    
    # Train optimized human classifier
    print(f"\n" + "="*60)
    print("👤 OPTIMIZED HUMAN CLASSIFICATION (Target: 90%+)")
    print("="*60)
    
    if len(np.unique(human_all_labels)) >= 2 and len(human_all_segments) > 10:
        human_classifier = EnhancedSTDPClassifier(n_input_features=human_all_segments.shape[1])
        
        X_human_train, X_human_test, y_human_train, y_human_test = train_test_split(
            human_all_segments, human_all_labels, test_size=0.25, random_state=42, stratify=human_all_labels
        )
        
        print(f"🎓 Training: {len(X_human_train)}, Testing: {len(X_human_test)}")
        
        human_history = human_classifier.train(X_human_train, y_human_train, n_epochs=60)
        human_predictions = human_classifier.predict(X_human_test)
        human_accuracy = accuracy_score(y_human_test, human_predictions)
        
        results['human'] = human_accuracy
        
        print(f"🎯 HUMAN RESULTS:")
        print(f"   📊 Test Accuracy: {human_accuracy:.1%}")
        print(f"   {'🎉 EXCELLENT!' if human_accuracy >= 0.90 else '👍 GOOD!' if human_accuracy >= 0.80 else '🔧 NEEDS IMPROVEMENT'}")
        
        with open('optimized_human_stdp_model.pkl', 'wb') as f:
            pickle.dump(human_classifier, f)
    else:
        print("❌ Insufficient human data")
        results['human'] = 0.0
    
    # Final results
    print(f"\n" + "="*80)
    print("🏆 FINAL OPTIMIZED RESULTS")
    print("="*80)
    print(f"🚗 Car Classification: {results['car']:.1%}")
    print(f"👤 Human Classification: {results['human']:.1%}")
    
    if results['car'] >= 0.90 and results['human'] >= 0.90:
        print("🎉 🎯 SUCCESS! Both classifiers achieved 90%+ accuracy!")
    elif results['car'] >= 0.80 or results['human'] >= 0.80:
        print("👍 Good progress! At least one classifier performing well.")
    else:
        print("🔧 Need further optimization.")
    
    return results

if __name__ == "__main__":
    try:
        results = run_optimized_classification()
        print(f"\n✅ Optimized classification completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
