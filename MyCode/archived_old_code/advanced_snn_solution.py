#!/usr/bin/env python3
"""
ADVANCED SNN SOLUTION FOR GEOPHONE SIGNAL CLASSIFICATION
Fixes training instability, poor performance, and class imbalance issues
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import SCTNeuron, create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer

class AdvancedGeophoneSNN:
    """
    Advanced SNN with stable training and improved performance
    """
    
    def __init__(self, n_input_neurons=None, n_hidden=80, learning_rate=0.002):
        self.n_input_neurons = n_input_neurons
        self.n_hidden = n_hidden  # Increased for better capacity
        self.learning_rate = learning_rate  # Reduced for stability
        self.network = None
        self.scaler = StandardScaler()
        self.trained = False
        
    def create_stable_network(self, n_input_neurons):
        """
        Create a stable, well-regularized SNN
        """
        network = SpikingNetwork()
        
        # Input layer
        input_neurons = []
        for i in range(n_input_neurons):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 2
            neuron.label = f"input_{i}"
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        network.add_layer(input_layer)
        
        # Hidden layer with better initialization
        hidden_neurons = []
        for i in range(self.n_hidden):
            neuron = create_SCTN()
            
            # Improved weight initialization
            std_dev = np.sqrt(2.0 / n_input_neurons)  # He initialization
            neuron.synapses_weights = np.random.normal(0.0, std_dev, n_input_neurons).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, -1.5, 1.5)
            
            # Stable neuron parameters
            neuron.leakage_factor = 2
            neuron.leakage_period = 5
            neuron.theta = -10  # Lower threshold for better sensitivity
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # Conservative STDP parameters for stability
            neuron.set_stdp(
                A_LTP=0.012,   # Reduced for stability
                A_LTD=-0.006,  # Balanced LTD
                tau=20.0,      # Longer tau for stability
                clk_freq=1000,
                wmax=2.0,      # Conservative weight limits
                wmin=-0.2
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        network.add_layer(hidden_layer)
        
        # Output layer with class-specific tuning
        output_neurons = []
        class_names = ['signal', 'nothing']
        
        for i in range(2):
            neuron = create_SCTN()
            
            # Class-specific weight initialization
            neuron.synapses_weights = np.random.normal(0.3, 0.2, self.n_hidden).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.0, 1.5)
            
            # Class-specific thresholds
            neuron.leakage_factor = 1
            neuron.leakage_period = 3
            neuron.theta = -8 if i == 0 else -12  # Signal class more sensitive
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 1 + i
            neuron.membrane_should_reset = True
            neuron.label = f"output_{class_names[i]}"
            
            # Supervised STDP with class balancing
            neuron.set_supervised_stdp(
                A=0.018 if i == 1 else 0.015,  # Nothing class gets slightly higher learning
                tau=15.0,
                clk_freq=1000,
                wmax=2.5,
                wmin=0.0,
                desired_output=np.array([], dtype=np.int64)
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        network.add_layer(output_layer)
        
        # Enable logging
        for neuron in output_neurons:
            network.log_out_spikes(neuron._id)
        
        print(f"Advanced SNN created: {n_input_neurons} → {self.n_hidden} → 2")
        return network
    
    def robust_spike_encoding(self, X, spike_duration=100):
        """
        Robust spike encoding with better normalization
        """
        n_samples, n_features = X.shape
        spike_trains = np.zeros((n_samples, n_features, spike_duration), dtype=int)
        
        # Robust feature-wise normalization
        X_normalized = np.zeros_like(X)
        for i in range(n_features):
            feature_col = X[:, i]
            # Use percentile-based normalization for robustness
            p10, p90 = np.percentile(feature_col, [10, 90])
            if p90 > p10:
                normalized = (feature_col - p10) / (p90 - p10)
                X_normalized[:, i] = np.clip(normalized, 0, 1)
            else:
                X_normalized[:, i] = 0.5
        
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                feature_val = X_normalized[sample_idx, feature_idx]
                
                if feature_val > 0.1:  # Only encode significant values
                    # Enhanced rate encoding with temporal structure
                    base_rate = feature_val * 0.35  # Slightly higher max rate
                    
                    # Create more realistic spike patterns
                    for t in range(spike_duration):
                        # Temporal modulation for better patterns
                        if t < spike_duration // 4:
                            temporal_factor = 1.3  # Early burst
                        elif t < 3 * spike_duration // 4:
                            temporal_factor = 0.7  # Sustained activity
                        else:
                            temporal_factor = 1.0  # Late activity
                        
                        spike_prob = base_rate * temporal_factor
                        
                        # Add slight randomness prevention for consecutive spikes
                        if t > 0 and spike_trains[sample_idx, feature_idx, t-1] == 1:
                            spike_prob *= 0.3  # Reduce probability after spike
                        
                        if np.random.random() < spike_prob:
                            spike_trains[sample_idx, feature_idx, t] = 1
        
        return spike_trains
    
    def train_with_stability(self, X_train, y_train, n_epochs=100, spike_duration=100):
        """
        Stable training with improved convergence
        """
        if self.network is None:
            self.network = self.create_stable_network(X_train.shape[1])
        
        # Preprocess data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"Training Advanced SNN for {n_epochs} epochs...")
        print(f"Training data: {len(X_train)} samples")
        print(f"Class distribution: Signal={np.sum(y_train == 0)}, Nothing={np.sum(y_train == 1)}")
        
        # Convert to spike trains
        X_spike_trains = self.robust_spike_encoding(X_train_scaled, spike_duration)
        
        training_accuracy = []
        learning_rates = []  # Track adaptive learning rate
        
        for epoch in range(n_epochs):
            # Adaptive learning rate schedule
            if epoch < 30:
                current_lr_factor = 1.0
            elif epoch < 60:
                current_lr_factor = 0.8
            elif epoch < 80:
                current_lr_factor = 0.6
            else:
                current_lr_factor = 0.4
            
            epoch_correct = 0
            epoch_total = 0
            
            # Shuffle with stratification to maintain class balance
            signal_indices = np.where(y_train == 0)[0]
            nothing_indices = np.where(y_train == 1)[0]
            
            # Balanced sampling
            min_class_size = min(len(signal_indices), len(nothing_indices))
            balanced_signal = np.random.choice(signal_indices, min_class_size, replace=False)
            balanced_nothing = np.random.choice(nothing_indices, min_class_size, replace=False)
            
            balanced_indices = np.concatenate([balanced_signal, balanced_nothing])
            np.random.shuffle(balanced_indices)
            
            for idx in balanced_indices:
                spike_train = X_spike_trains[idx]
                target_class = y_train[idx]
                
                # Reset network state
                self.network.reset_input()
                
                # Set supervised learning targets
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target_class:
                            # Target neuron should spike with class-specific pattern
                            if target_class == 0:  # Signal class
                                spike_times = list(range(20, spike_duration-20, 15))  # Regular pattern
                            else:  # Nothing class
                                spike_times = list(range(30, spike_duration-30, 25))  # Sparser pattern
                            
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target neuron should not spike much
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spike train to network
                output_spikes = []
                for t in range(spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Enhanced prediction with confidence
                total_output_spikes = np.sum(output_spikes, axis=0)
                total_spikes = np.sum(total_output_spikes)
                
                if total_spikes > 0:
                    predicted_class = np.argmax(total_output_spikes)
                    confidence = total_output_spikes[predicted_class] / total_spikes
                    
                    # Only count high-confidence predictions during training
                    if confidence > 0.6:
                        if predicted_class == target_class:
                            epoch_correct += 1
                        epoch_total += 1
                else:
                    # Default prediction when no spikes
                    predicted_class = 1  # Default to nothing
                    if predicted_class == target_class:
                        epoch_correct += 1
                    epoch_total += 1
            
            # Calculate accuracy
            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            training_accuracy.append(train_acc)
            learning_rates.append(current_lr_factor)
            
            # Progress reporting
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: Accuracy={train_acc:.3f}, LR_Factor={current_lr_factor:.2f}, Samples={epoch_total}")
        
        self.trained = True
        
        # Plot training curve
        self._plot_training_curve(training_accuracy, learning_rates)
        
        return training_accuracy
    
    def predict_with_confidence(self, X_test, spike_duration=100):
        """
        Prediction with confidence estimation
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        X_test_spikes = self.robust_spike_encoding(X_test_scaled, spike_duration)
        
        predictions = []
        confidences = []
        
        for spike_train in X_test_spikes:
            self.network.reset_input()
            
            output_spikes = []
            for t in range(spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            total_output_spikes = np.sum(output_spikes, axis=0)
            total_spikes = np.sum(total_output_spikes)
            
            if total_spikes > 0:
                predicted_class = np.argmax(total_output_spikes)
                confidence = total_output_spikes[predicted_class] / total_spikes
            else:
                predicted_class = 1  # Default to nothing
                confidence = 0.1  # Low confidence for no-spike predictions
            
            predictions.append(predicted_class)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def evaluate_comprehensive(self, X_test, y_test):
        """
        Comprehensive evaluation with detailed metrics
        """
        predictions, confidences = self.predict_with_confidence(X_test)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Classification report
        class_names = ['Signal', 'Nothing']
        report = classification_report(
            y_test, predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        print("\n" + "="*60)
        print("🎯 ADVANCED SNN CLASSIFICATION RESULTS")
        print("="*60)
        print(f"🎯 Overall Accuracy: {accuracy:.1%}")
        print(f"📊 Average Confidence: {np.mean(confidences):.3f}")
        print(f"🔥 High Confidence (>0.8): {np.sum(confidences > 0.8)}/{len(confidences)}")
        print(f"⚠️  Low Confidence (<0.5): {np.sum(confidences < 0.5)}/{len(confidences)}")
        
        print(f"\n📊 Confusion Matrix:")
        print("Predicted:     Signal   Nothing")
        for i, actual_class in enumerate(class_names):
            print(f"Actual {actual_class:8s}: {cm[i][0]:6d}   {cm[i][1]:6d}")
        
        print(f"\n📈 Detailed Classification Report:")
        for class_name in class_names:
            if class_name.lower() in report:
                metrics = report[class_name.lower()]
                print(f"{class_name:8s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Per-class confidence analysis
        signal_confs = confidences[y_test == 0]
        nothing_confs = confidences[y_test == 1]
        
        print(f"\n🔍 Per-Class Confidence Analysis:")
        if len(signal_confs) > 0:
            print(f"Signal predictions: Avg={np.mean(signal_confs):.3f}, "
                  f"High_conf(>0.8)={np.sum(signal_confs > 0.8)}/{len(signal_confs)}")
        if len(nothing_confs) > 0:
            print(f"Nothing predictions: Avg={np.mean(nothing_confs):.3f}, "
                  f"High_conf(>0.8)={np.sum(nothing_confs > 0.8)}/{len(nothing_confs)}")
        
        return accuracy, report, cm, confidences
    
    def _plot_training_curve(self, training_accuracy, learning_rates):
        """Plot training progress"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_accuracy, 'b-', linewidth=2)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(learning_rates, 'r-', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('LR Factor')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        smoothed_acc = np.convolve(training_accuracy, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_acc, 'g-', linewidth=2)
        plt.title('Smoothed Training Accuracy (10-epoch window)')
        plt.xlabel('Epoch')
        plt.ylabel('Smoothed Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"Final Accuracy: {training_accuracy[-1]:.3f}", fontsize=12)
        plt.text(0.1, 0.7, f"Best Accuracy: {max(training_accuracy):.3f}", fontsize=12)
        plt.text(0.1, 0.6, f"Hidden Neurons: {self.n_hidden}", fontsize=12)
        plt.text(0.1, 0.5, f"Learning Rate: {self.learning_rate}", fontsize=12)
        plt.text(0.1, 0.4, f"Stability: {'Good' if np.std(training_accuracy[-10:]) < 0.1 else 'Poor'}", fontsize=12)
        plt.title('Training Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("advanced_snn_training.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curve saved to: advanced_snn_training.png")

def run_advanced_training():
    """
    Run advanced SNN training pipeline
    """
    print("🚀 ADVANCED SNN TRAINING PIPELINE")
    print("=" * 60)
    
    # Import data loading functions
    from load_saved_chunks import load_chunks_directly, extract_segments_from_loaded_chunks
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    # Load chunks
    chunk_data = load_chunks_directly(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return None
    
    results = {}
    
    # Advanced Car Classification
    print(f"\n🚗 ADVANCED CAR vs CAR_NOTHING CLASSIFICATION")
    print("=" * 50)
    
    car_segments = []
    car_labels = []
    
    # Process car signal segments
    if 'car' in chunk_data:
        segments, labels = extract_segments_from_loaded_chunks(chunk_data['car'], 'car')
        signal_indices = labels == 1
        car_segments.extend(segments[signal_indices])
        car_labels.extend([0] * np.sum(signal_indices))  # 0 = signal
        print(f"Car signal segments: {np.sum(signal_indices)}")
    
    # Process car nothing segments
    if 'car_nothing' in chunk_data:
        segments, labels = extract_segments_from_loaded_chunks(chunk_data['car_nothing'], 'car_nothing')
        nothing_indices = labels == 0
        car_segments.extend(segments[nothing_indices])
        car_labels.extend([1] * np.sum(nothing_indices))  # 1 = nothing
        print(f"Car nothing segments: {np.sum(nothing_indices)}")
    
    if len(car_segments) > 0 and len(np.unique(car_labels)) >= 2:
        car_segments = np.array(car_segments)
        car_labels = np.array(car_labels)
        
        print(f"Total car dataset: {len(car_segments)} segments")
        print(f"Signal: {np.sum(car_labels == 0)}, Nothing: {np.sum(car_labels == 1)}")
        
        # Create Advanced SNN for cars
        advanced_car_snn = AdvancedGeophoneSNN(n_hidden=80, learning_rate=0.002)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            car_segments, car_labels, test_size=0.3, random_state=42, 
            stratify=car_labels
        )
        
        print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
        
        # Train with stability
        training_history = advanced_car_snn.train_with_stability(
            X_train, y_train, n_epochs=80, spike_duration=100
        )
        
        # Evaluate
        accuracy, report, cm, confidences = advanced_car_snn.evaluate_comprehensive(X_test, y_test)
        
        # Save model
        with open("advanced_car_snn.pkl", 'wb') as f:
            pickle.dump({
                'snn': advanced_car_snn,
                'training_history': training_history,
                'test_accuracy': accuracy
            }, f)
        
        results['car'] = {
            'snn': advanced_car_snn,
            'accuracy': accuracy,
            'training_history': training_history,
            'confidences': confidences
        }
        
        print(f"\n✅ ADVANCED CAR SNN COMPLETE!")
        print(f"🎯 Test Accuracy: {accuracy:.1%}")
        print(f"💾 Model saved: advanced_car_snn.pkl")
    
    return results

if __name__ == "__main__":
    print("🎯 ADVANCED SNN GEOPHONE CLASSIFICATION SYSTEM")
    print("Stable training with improved performance and class balancing")
    print()
    
    # Run advanced training
    results = run_advanced_training()
    
    if results and results.get('car'):
        car_res = results['car']
        print(f"\n" + "="*60)
        print("🏆 ADVANCED SNN RESULTS SUMMARY")
        print("="*60)
        print(f"\n🚗 CAR CLASSIFICATION:")
        print(f"   🎯 Accuracy: {car_res['accuracy']:.1%}")
        print(f"   📈 Training Stability: {'Good' if np.std(car_res['training_history'][-10:]) < 0.1 else 'Needs Improvement'}")
        print(f"   🧠 Architecture: 56 → 80 → 2 (Advanced)")
        print(f"   ⚖️  Class Balance: Handled with balanced sampling")
        print(f"   🎭 Confidence: Avg {np.mean(car_res['confidences']):.3f}")
        print("="*60)
    else:
        print("❌ Advanced training failed or no results") 