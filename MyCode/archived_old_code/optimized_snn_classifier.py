#!/usr/bin/env python3
# Optimized SNN Classifier for Geophone Signals
# Improved architecture and training stability

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

# Import sctnN components
from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import SCTNeuron, create_SCTN, IDENTITY, BINARY
from sctnN.layers import SCTNLayer
from sctnN.learning_rules.stdp import STDP
from sctnN.learning_rules.supervised_stdp import SupervisedSTDP

# Import our new advanced pattern analyzer
from advanced_pattern_analyzer import AdvancedPatternAnalyzer

class OptimizedGeophoneSNN:
    """
    Optimized Spiking Neural Network for geophone signal classification
    Improved architecture, training stability, and performance
    """
    
    def __init__(self, n_input_neurons=None, n_hidden=60, learning_rate=0.008, architecture='balanced'):
        self.n_input_neurons = n_input_neurons
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.network = None
        self.scaler = RobustScaler()
        self.trained = False
        self.training_history = []
        
        # Optimized parameters for stability
        self.spike_encoding_params = {
            'duration': 150,  # Reduced from 200 for faster training
            'encoding_type': 'temporal_rate',
            'noise_level': 0.05
        }
        
        # Class balancing
        self.class_weights = None
        self.balance_classes = True
        
        # Classification labels
        self.class_labels = {0: 'signal', 1: 'nothing'}
        
    def create_optimized_network(self, n_input_neurons):
        """
        Create optimized SNN architecture with improved stability
        """
        self.n_input_neurons = n_input_neurons
        
        # Create the spiking network
        self.network = SpikingNetwork()
        
        # Input layer (feature inputs)
        input_neurons = []
        for i in range(n_input_neurons):
            neuron = create_SCTN()
            neuron.activation_function = IDENTITY
            neuron.threshold_pulse = 8  # Reduced threshold for better sensitivity
            neuron.leakage_factor = 1
            neuron.leakage_period = 2
            neuron.label = f"input_{i}"
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        
        # Hidden layer with optimized parameters
        hidden_neurons = []
        for i in range(self.n_hidden):
            neuron = create_SCTN()
            
            # Improved weight initialization (Xavier-like)
            fan_in = n_input_neurons
            weight_scale = np.sqrt(2.0 / fan_in)
            neuron.synapses_weights = np.random.normal(0.5, weight_scale, n_input_neurons).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.1, 2.0)
            
            # Optimized neuron parameters
            neuron.leakage_factor = 1
            neuron.leakage_period = 4  # Increased for stability
            neuron.theta = -8  # Improved threshold
            neuron.activation_function = IDENTITY
            neuron.membrane_should_reset = True
            neuron.label = f"hidden_{i}"
            
            # Improved STDP parameters
            neuron.set_stdp(
                A_LTP=0.015,  # Reduced learning rate for stability
                A_LTD=-0.008,  # Balanced LTD
                tau=20.0,     # Increased time constant
                clk_freq=1000,
                wmax=2.5,     # Reduced max weight
                wmin=0.05     # Increased min weight
            )
            
            hidden_neurons.append(neuron)
        
        hidden_layer = SCTNLayer(hidden_neurons)
        self.network.add_layer(hidden_layer)
        
        # Output layer with optimized parameters
        output_neurons = []
        class_names = ['signal', 'nothing']
        for i in range(2):
            neuron = create_SCTN()
            
            # Improved weight initialization for output layer
            fan_in = self.n_hidden
            weight_scale = np.sqrt(1.0 / fan_in)
            neuron.synapses_weights = np.random.normal(0.8, weight_scale, self.n_hidden).astype(np.float64)
            neuron.synapses_weights = np.clip(neuron.synapses_weights, 0.2, 2.0)
            
            # Output neuron parameters
            neuron.leakage_factor = 2
            neuron.leakage_period = 6
            neuron.theta = -12  # Higher threshold for output decisions
            neuron.activation_function = BINARY
            neuron.threshold_pulse = 10
            neuron.membrane_should_reset = True
            neuron.label = f"output_{class_names[i]}"
            
            # Supervised STDP for classification
            neuron.set_supervised_stdp(
                A=0.020,      # Adjusted learning rate
                tau=15.0,     # Improved time constant
                clk_freq=1000,
                wmax=2.5,
                wmin=0.05,
                desired_output=np.array([], dtype=np.int64)
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        
        # Enable spike logging for output neurons
        for neuron in output_neurons:
            self.network.log_out_spikes(neuron._id)
        
        print(f"Optimized SNN created: {n_input_neurons} input → {self.n_hidden} hidden → 2 output neurons")
        print(f"Architecture: {self.architecture}, Learning rate: {self.learning_rate}")
        return self.network
    
    def preprocess_features(self, X, fit_scaler=False):
        """
        Robust feature preprocessing
        """
        X = np.array(X, dtype=np.float64)
        
        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Robust scaling
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Ensure positive values for spike encoding
        X_scaled = X_scaled - np.min(X_scaled, axis=0) + 0.1
        
        return X_scaled
    
    def create_optimized_spike_encoding(self, X, spike_duration=150):
        """
        Create optimized spike encoding with temporal patterns
        """
        n_samples, n_features = X.shape
        spike_trains = np.zeros((n_samples, n_features, spike_duration), dtype=np.int32)
        
        for i in range(n_samples):
            for j in range(n_features):
                feature_value = X[i, j]
                
                # Temporal rate encoding with improved distribution
                if feature_value > 0:
                    # Base rate proportional to feature value
                    base_rate = min(feature_value * 0.3, 0.8)  # Max 80% spike probability
                    
                    # Add temporal structure
                    for t in range(spike_duration):
                        # Early spikes for strong features
                        temporal_modulation = 1.0 - (t / spike_duration) * 0.3
                        spike_prob = base_rate * temporal_modulation
                        
                        # Add small amount of noise for robustness
                        noise = np.random.uniform(-0.05, 0.05)
                        final_prob = max(0, min(1, spike_prob + noise))
                        
                        spike_trains[i, j, t] = 1 if np.random.random() < final_prob else 0
        
        return spike_trains
    
    def train_with_cross_validation(self, X, y, n_epochs=80, cv_folds=3):
        """
        Train with cross-validation for better generalization
        """
        if self.network is None:
            self.create_optimized_network(X.shape[1])
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit_scaler=True)
        
        # Calculate class weights for balancing
        if self.balance_classes:
            self.class_weights = compute_class_weight(
                'balanced', classes=np.unique(y), y=y
            )
            print(f"Class weights: {self.class_weights}")
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        print(f"Training with {cv_folds}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
            print(f"\\nFold {fold + 1}/{cv_folds}:")
            
            X_train_fold, X_val_fold = X_processed[train_idx], X_processed[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train on this fold
            fold_history = self._train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                n_epochs=n_epochs, fold_id=fold
            )
            
            # Validate on this fold
            val_predictions, _ = self.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, val_predictions, average='weighted')
            cv_scores.append(fold_score)
            
            print(f"Fold {fold + 1} F1 Score: {fold_score:.3f}")
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"\\nCross-validation results:")
        print(f"Mean F1 Score: {mean_cv_score:.3f} ± {std_cv_score:.3f}")
        
        # Final training on full dataset
        print(f"\\nFinal training on full dataset...")
        self.training_history = self._train_single_fold(
            X_processed, y, X_processed, y,  # Use full dataset for final training
            n_epochs=n_epochs, fold_id='final'
        )
        
        self.trained = True
        return self.training_history, cv_scores
    
    def _train_single_fold(self, X_train, y_train, X_val, y_val, n_epochs=80, fold_id=0):
        """
        Train a single fold with improved stability
        """
        # Convert to spike trains
        X_spike_trains = self.create_optimized_spike_encoding(
            X_train, self.spike_encoding_params['duration']
        )
        
        fold_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': []
        }
        
        spike_duration = self.spike_encoding_params['duration']
        
        # Adaptive learning rate schedule
        initial_lr = self.learning_rate
        
        for epoch in range(n_epochs):
            # Adaptive learning rate
            if epoch > 0 and epoch % 20 == 0:
                self.learning_rate *= 0.9  # Gradual decay
            
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_spike_trains))
            
            for idx in indices:
                spike_train = X_spike_trains[idx]
                target_class = y_train[idx]
                
                # Apply class weighting
                sample_weight = self.class_weights[target_class] if self.class_weights is not None else 1.0
                
                # Reset network state
                self.network.reset_input()
                
                # Set desired output with improved patterns
                for neuron_idx, neuron in enumerate(self.network.layers_neurons[-1].neurons):
                    if hasattr(neuron, 'supervised_stdp') and neuron.supervised_stdp is not None:
                        if neuron_idx == target_class:
                            # Target neuron should spike with temporal pattern
                            spike_times = list(range(15, spike_duration-15, 25))  # Distributed spikes
                            # Add class weighting by repeating spikes
                            if sample_weight > 1.0:
                                additional_spikes = list(range(20, spike_duration-20, 30))
                                spike_times.extend(additional_spikes)
                            neuron.supervised_stdp.desired_output = np.array(spike_times, dtype=np.int64)
                        else:
                            # Non-target neuron should not spike
                            neuron.supervised_stdp.desired_output = np.array([-1], dtype=np.int64)
                
                # Present spike train to network
                output_spikes = []
                for t in range(spike_duration):
                    input_spikes = spike_train[:, t]
                    output = self.network.input(input_spikes)
                    output_spikes.append(output)
                
                # Calculate prediction and loss
                total_output_spikes = np.sum(output_spikes, axis=0)
                predicted_class = np.argmax(total_output_spikes)
                
                # Simple loss calculation
                correct_prediction = (predicted_class == target_class)
                loss = 0 if correct_prediction else 1
                epoch_loss += loss
                
                if correct_prediction:
                    epoch_correct += 1
                epoch_total += 1
            
            # Calculate training metrics
            train_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            train_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
            
            # Validation metrics
            if len(X_val) > 0 and epoch % 5 == 0:  # Validate every 5 epochs
                val_predictions, _ = self.predict(X_val)
                val_accuracy = np.mean(val_predictions == y_val)
                val_loss = np.mean(val_predictions != y_val)
            else:
                val_accuracy = train_accuracy
                val_loss = train_loss
            
            # Store history
            fold_history['train_accuracy'].append(train_accuracy)
            fold_history['val_accuracy'].append(val_accuracy)
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:3d}: Train Acc={train_accuracy:.3f}, Val Acc={val_accuracy:.3f}, LR={self.learning_rate:.6f}")
        
        # Reset learning rate
        self.learning_rate = initial_lr
        
        return fold_history
    
    def predict(self, X_test):
        """
        Make predictions with confidence scores
        """
        if not self.trained:
            raise ValueError("Network must be trained before making predictions")
        
        # Preprocess features
        X_processed = self.preprocess_features(X_test, fit_scaler=False)
        
        # Convert to spike trains
        X_spike_trains = self.create_optimized_spike_encoding(
            X_processed, self.spike_encoding_params['duration']
        )
        
        predictions = []
        confidence_scores = []
        
        spike_duration = self.spike_encoding_params['duration']
        
        for spike_train in X_spike_trains:
            # Reset network state
            self.network.reset_input()
            
            # Present spike train to network
            output_spikes = []
            for t in range(spike_duration):
                input_spikes = spike_train[:, t]
                output = self.network.input(input_spikes)
                output_spikes.append(output)
            
            # Calculate prediction and confidence
            total_output_spikes = np.sum(output_spikes, axis=0)
            predicted_class = np.argmax(total_output_spikes)
            
            # Improved confidence calculation
            total_spikes = np.sum(total_output_spikes)
            if total_spikes > 0:
                confidence = total_output_spikes[predicted_class] / total_spikes
                # Add certainty bonus based on spike difference
                spike_diff = np.abs(total_output_spikes[0] - total_output_spikes[1])
                certainty_bonus = min(0.2, spike_diff / (total_spikes + 1))
                confidence = min(1.0, confidence + certainty_bonus)
            else:
                confidence = 0.5  # Uncertain when no spikes
            
            predictions.append(predicted_class)
            confidence_scores.append(confidence)
        
        return np.array(predictions), np.array(confidence_scores)
    
    def evaluate_comprehensive(self, X_test, y_test):
        """
        Comprehensive evaluation with detailed metrics
        """
        predictions, confidence = self.predict(X_test)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Classification report
        class_names = ['Signal', 'Nothing']
        report = classification_report(
            y_test, predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Additional metrics
        avg_confidence = np.mean(confidence)
        high_confidence_mask = confidence > 0.7
        high_conf_accuracy = np.mean(predictions[high_confidence_mask] == y_test[high_confidence_mask]) if np.sum(high_confidence_mask) > 0 else 0
        
        print("\\n" + "="*80)
        print("OPTIMIZED SNN CLASSIFICATION RESULTS")
        print("="*80)
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Weighted F1 Score: {f1:.3f}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"High-Confidence Accuracy (>{0.7:.1f}): {high_conf_accuracy:.3f} ({np.sum(high_confidence_mask)} samples)")
        
        print("\\nConfusion Matrix:")
        print("              Predicted")
        print("Actual    Signal  Nothing")
        for i, actual_class in enumerate(class_names):
            print(f"{actual_class:8s}: {cm[i][0]:6d}  {cm[i][1]:6d}")
        
        print("\\nDetailed Classification Report:")
        for class_name in class_names:
            class_key = class_name.lower()
            if class_key in report:
                metrics = report[class_key]
                print(f"{class_name:8s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                      f"Support={metrics['support']}")
        
        # Class-specific confidence analysis
        for class_idx, class_name in enumerate(class_names):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_confidence = confidence[class_mask]
                class_accuracy = np.mean(predictions[class_mask] == y_test[class_mask])
                print(f"{class_name} class: Avg confidence={np.mean(class_confidence):.3f}, "
                      f"Accuracy={class_accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'confidence': confidence,
            'avg_confidence': avg_confidence,
            'high_conf_accuracy': high_conf_accuracy
        }
    
    def save_model(self, filepath):
        """Save the trained SNN model with all components"""
        model_data = {
            'network_weights': self.get_network_weights(),
            'scaler': self.scaler,
            'n_input_neurons': self.n_input_neurons,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'architecture': self.architecture,
            'trained': self.trained,
            'training_history': self.training_history,
            'spike_encoding_params': self.spike_encoding_params,
            'class_weights': self.class_weights,
            'class_labels': self.class_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Optimized SNN model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved SNN model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model state
        self.scaler = model_data['scaler']
        self.n_input_neurons = model_data['n_input_neurons']
        self.n_hidden = model_data['n_hidden']
        self.learning_rate = model_data['learning_rate']
        self.architecture = model_data.get('architecture', 'balanced')
        self.trained = model_data['trained']
        self.training_history = model_data.get('training_history', [])
        self.spike_encoding_params = model_data.get('spike_encoding_params', self.spike_encoding_params)
        self.class_weights = model_data.get('class_weights', None)
        self.class_labels = model_data.get('class_labels', self.class_labels)
        
        # Recreate network and restore weights
        if self.n_input_neurons is not None:
            self.create_optimized_network(self.n_input_neurons)
            self.set_network_weights(model_data['network_weights'])
        
        print(f"Optimized SNN model loaded from {filepath}")
    
    def get_network_weights(self):
        """Extract weights from the network for saving"""
        weights = {}
        if self.network is not None:
            for layer_idx, layer in enumerate(self.network.layers_neurons):
                layer_weights = []
                for neuron in layer.neurons:
                    layer_weights.append({
                        'synapses_weights': neuron.synapses_weights.copy() if hasattr(neuron, 'synapses_weights') else None,
                        'theta': neuron.theta,
                        'leakage_factor': neuron.leakage_factor,
                        'leakage_period': neuron.leakage_period
                    })
                weights[f'layer_{layer_idx}'] = layer_weights
        return weights
    
    def set_network_weights(self, weights):
        """Restore weights to the network"""
        if self.network is not None:
            for layer_idx, layer in enumerate(self.network.layers_neurons):
                layer_key = f'layer_{layer_idx}'
                if layer_key in weights:
                    layer_weights = weights[layer_key]
                    for neuron_idx, neuron in enumerate(layer.neurons):
                        if neuron_idx < len(layer_weights):
                            neuron_data = layer_weights[neuron_idx]
                            if 'synapses_weights' in neuron_data and neuron_data['synapses_weights'] is not None:
                                neuron.synapses_weights = neuron_data['synapses_weights'].copy()
                            neuron.theta = neuron_data['theta']
                            neuron.leakage_factor = neuron_data['leakage_factor']
                            neuron.leakage_period = neuron_data['leakage_period']
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.training_history:
            print("No training history available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.training_history['train_accuracy']))
        
        # Training and validation accuracy
        ax1.plot(epochs, self.training_history['train_accuracy'], label='Training', linewidth=2)
        ax1.plot(epochs, self.training_history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training and validation loss
        ax2.plot(epochs, self.training_history['train_loss'], label='Training', linewidth=2)
        ax2.plot(epochs, self.training_history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning progress
        window_size = max(1, len(epochs) // 20)
        if len(epochs) > window_size:
            smoothed_train = np.convolve(self.training_history['train_accuracy'], 
                                       np.ones(window_size)/window_size, mode='valid')
            smoothed_val = np.convolve(self.training_history['val_accuracy'], 
                                     np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = range(window_size-1, len(epochs))
            
            ax3.plot(smooth_epochs, smoothed_train, label='Smoothed Training', linewidth=2)
            ax3.plot(smooth_epochs, smoothed_val, label='Smoothed Validation', linewidth=2)
        ax3.set_title('Smoothed Learning Progress')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Final statistics
        final_train_acc = self.training_history['train_accuracy'][-1]
        final_val_acc = self.training_history['val_accuracy'][-1]
        best_val_acc = max(self.training_history['val_accuracy'])
        
        ax4.bar(['Final Train', 'Final Val', 'Best Val'], 
                [final_train_acc, final_val_acc, best_val_acc],
                color=['blue', 'orange', 'green'], alpha=0.7)
        ax4.set_title('Final Performance')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    print("🧠 Optimized Geophone SNN Classifier")
    print("Features: Improved architecture, training stability, cross-validation")
    
    # Create test instance
    snn = OptimizedGeophoneSNN(n_hidden=60, learning_rate=0.008)
    print(f"✅ Created optimized SNN with {snn.n_hidden} hidden neurons")
