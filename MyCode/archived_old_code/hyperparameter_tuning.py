import numpy as np
import pandas as pd
import time
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import your existing components
import sys
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY
from sctnN.spiking_network import SpikingNetwork
from sctnN.layers import SCTNLayer

# Import from your existing resonator_work.py
from resonator_work import ProductionFeatureExtractor, CHUNKED_OUTPUT_DIR

class MultiLayerSCTNClassifier:
    """Multi-layer SCTN network with configurable architecture and hyperparameters"""
    
    def __init__(self, input_size=16, hidden_layers=[20], output_size=1, 
                 learning_rate=0.1, signal_type=None, 
                 threshold_range=(20.0, 60.0), weight_init_std=0.1,
                 stdp_params=None):
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers  # List of neurons per hidden layer
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.signal_type = signal_type
        self.threshold_range = threshold_range
        self.weight_init_std = weight_init_std
        
        # Default STDP parameters
        self.stdp_params = stdp_params or {
            'A_LTP': 0.01,
            'A_LTD': 0.005,
            'tau': 0.02,
            'wmax': 1.0,
            'wmin': 0.0
        }
        
        self.network = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        
    def _create_network(self):
        """Create multi-layer SCTN network"""
        self.network = SpikingNetwork()
        all_layers = []
        
        # Input layer
        input_neurons = []
        for i in range(self.input_size):
            neuron = create_SCTN()
            neuron.label = f"input_{i}"
            neuron.activation_function = BINARY
            input_neurons.append(neuron)
        
        input_layer = SCTNLayer(input_neurons)
        self.network.add_layer(input_layer)
        all_layers.append(input_layer)
        
        # Hidden layers
        prev_layer_size = self.input_size
        for layer_idx, layer_size in enumerate(self.hidden_layers):
            hidden_neurons = []
            
            for i in range(layer_size):
                neuron = create_SCTN()
                neuron.label = f"hidden_{layer_idx}_{i}"
                
                # Initialize weights
                neuron.synapses_weights = np.random.normal(
                    0, self.weight_init_std, prev_layer_size
                ).astype(np.float64)
                
                # Set threshold
                neuron.threshold_pulse = np.random.uniform(
                    self.threshold_range[0], self.threshold_range[1]
                )
                
                neuron.activation_function = BINARY
                neuron.theta = 0.0
                neuron.reset_to = 0.0
                neuron.membrane_should_reset = True
                
                # Set STDP parameters
                neuron.set_stdp(
                    A_LTP=self.stdp_params['A_LTP'],
                    A_LTD=self.stdp_params['A_LTD'],
                    tau=self.stdp_params['tau'],
                    clk_freq=153600,
                    wmax=self.stdp_params['wmax'],
                    wmin=self.stdp_params['wmin']
                )
                
                hidden_neurons.append(neuron)
            
            hidden_layer = SCTNLayer(hidden_neurons)
            self.network.add_layer(hidden_layer)
            all_layers.append(hidden_layer)
            prev_layer_size = layer_size
        
        # Output layer
        output_neurons = []
        for i in range(self.output_size):
            neuron = create_SCTN()
            neuron.label = f"output_{i}"
            
            # Initialize weights
            neuron.synapses_weights = np.random.normal(
                0, self.weight_init_std, prev_layer_size
            ).astype(np.float64)
            
            # Set threshold
            neuron.threshold_pulse = np.random.uniform(
                self.threshold_range[0], self.threshold_range[1]
            )
            
            neuron.activation_function = BINARY
            neuron.theta = 0.0
            neuron.reset_to = 0.0
            neuron.membrane_should_reset = True
            
            # Set STDP parameters
            neuron.set_stdp(
                A_LTP=self.stdp_params['A_LTP'],
                A_LTD=self.stdp_params['A_LTD'],
                tau=self.stdp_params['tau'],
                clk_freq=153600,
                wmax=self.stdp_params['wmax'],
                wmin=self.stdp_params['wmin']
            )
            
            output_neurons.append(neuron)
        
        output_layer = SCTNLayer(output_neurons)
        self.network.add_layer(output_layer)
        all_layers.append(output_layer)
        
        return all_layers
    
    def _forward_pass(self, features):
        """Forward pass through the network"""
        # Reset all neurons
        for layer in self.network.layers:
            for neuron in layer.neurons:
                neuron.membrane_potential = 0.0
                neuron.index = 0
        
        # Input layer
        input_layer = self.network.layers[0]
        for i, neuron in enumerate(input_layer.neurons):
            if i < len(features):
                neuron.membrane_potential = features[i]
        
        # Process through all layers
        current_activations = features[:len(input_layer.neurons)]
        
        for layer_idx in range(1, len(self.network.layers)):
            layer = self.network.layers[layer_idx]
            next_activations = []
            
            for neuron in layer.neurons:
                # Compute weighted sum
                if len(current_activations) == len(neuron.synapses_weights):
                    activation = np.dot(current_activations, neuron.synapses_weights)
                else:
                    # Handle size mismatch
                    min_size = min(len(current_activations), len(neuron.synapses_weights))
                    activation = np.dot(current_activations[:min_size], neuron.synapses_weights[:min_size])
                
                neuron.membrane_potential = activation
                output = neuron._activation_function_binary()
                next_activations.append(output)
            
            current_activations = next_activations
        
        # Return output layer activation
        return current_activations[0] if len(current_activations) > 0 else 0
    
    def train(self, X, y, epochs=100, verbose=True):
        """Train the multi-layer SCTN network"""
        if verbose:
            print(f"🧠 Training Multi-Layer SCTN Network...")
            print(f"   Architecture: {self.input_size} → {' → '.join(map(str, self.hidden_layers))} → {self.output_size}")
            print(f"   Dataset: {len(X)} samples, {X.shape[1]} features")
            print(f"   Labels: {Counter(y)}")
        
        # Feature scaling
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler.fit_transform(X)
        
        # Create network
        self._create_network()
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        if verbose:
            print(f"   Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
            print(f"   Learning rate: {self.learning_rate}, Epochs: {epochs}")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            correct = 0
            
            for i in range(len(X_train)):
                features = X_train[i]
                target = y_train[i]
                
                # Forward pass
                prediction = self._forward_pass(features)
                
                # Simple learning rule (update output layer weights)
                error = target - prediction
                output_layer = self.network.layers[-1]
                
                for neuron in output_layer.neurons:
                    # Get activations from previous layer
                    if len(self.network.layers) > 1:
                        prev_layer = self.network.layers[-2]
                        prev_activations = []
                        for prev_neuron in prev_layer.neurons:
                            prev_activations.append(prev_neuron.membrane_potential)
                        
                        # Update weights
                        if len(prev_activations) == len(neuron.synapses_weights):
                            neuron.synapses_weights += self.learning_rate * error * np.array(prev_activations)
                    else:
                        # Direct connection from input
                        neuron.synapses_weights += self.learning_rate * error * features
                    
                    # Update threshold
                    neuron.threshold_pulse += self.learning_rate * error * 0.1
                
                if prediction == target:
                    correct += 1
            
            train_accuracy = correct / len(X_train)
            
            # Validation
            val_correct = 0
            for i in range(len(X_val)):
                features = X_val[i]
                target = y_val[i]
                prediction = self._forward_pass(features)
                if prediction == target:
                    val_correct += 1
            
            val_accuracy = val_correct / len(X_val)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy
            })
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:3d}: Train={train_accuracy:.4f}, Val={val_accuracy:.4f}")
        
        self.is_trained = True
        
        if verbose:
            print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return best_val_acc
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for features in X_scaled:
            prediction = self._forward_pass(features)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def get_architecture_string(self):
        """Get string representation of architecture"""
        return f"{self.input_size}-{'-'.join(map(str, self.hidden_layers))}-{self.output_size}"

class HyperparameterTuner:
    """Comprehensive hyperparameter tuning for SCTN networks using nested loops"""
    
    def __init__(self, X, y, signal_type='unknown'):
        self.X = X
        self.y = y
        self.signal_type = signal_type
        self.results = []
        
    def run_nested_grid_search(self):
        """Run nested loops to test all hyperparameter combinations"""
        print(f"🔍 NESTED HYPERPARAMETER SEARCH FOR {self.signal_type.upper()}")
        print("=" * 80)
        
        # Define hyperparameter ranges
        architectures = [
            [10],           # Single layer
            [20], 
            [30],
            [15, 10],       # Two layers
            [20, 15],
            [25, 20],
            [30, 15],
            [20, 10, 5],    # Three layers
            [25, 15, 8],
            [30, 20, 10],
        ]
        
        learning_rates = [0.05, 0.1, 0.15, 0.2]
        weight_stds = [0.05, 0.1, 0.15]
        threshold_ranges = [(15, 45), (20, 60), (25, 75)]
        epochs_list = [50, 75, 100]
        
        total_combinations = (len(architectures) * len(learning_rates) * 
                            len(weight_stds) * len(threshold_ranges) * len(epochs_list))
        
        print(f"📊 Total combinations to test: {total_combinations}")
        print("🔄 Starting nested loop search...")
        print()
        
        best_accuracy = 0
        best_config = None
        combination_count = 0
        
        # NESTED LOOPS START HERE
        for arch_idx, hidden_layers in enumerate(architectures):
            print(f"\n🏗️  Testing Architecture {arch_idx + 1}/{len(architectures)}: {hidden_layers}")
            
            for lr_idx, learning_rate in enumerate(learning_rates):
                print(f"  📈 Learning Rate {lr_idx + 1}/{len(learning_rates)}: {learning_rate}")
                
                for ws_idx, weight_std in enumerate(weight_stds):
                    print(f"    ⚖️  Weight Std {ws_idx + 1}/{len(weight_stds)}: {weight_std}")
                    
                    for tr_idx, threshold_range in enumerate(threshold_ranges):
                        print(f"      🎯 Threshold Range {tr_idx + 1}/{len(threshold_ranges)}: {threshold_range}")
                        
                        for ep_idx, epochs in enumerate(epochs_list):
                            combination_count += 1
                            print(f"        ⏱️  Epochs {ep_idx + 1}/{len(epochs_list)}: {epochs} "
                                  f"[Combo {combination_count}/{total_combinations}]")
                            
                            try:
                                # Cross-validation
                                cv_accuracies = []
                                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                                
                                for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
                                    X_train_fold = self.X[train_idx]
                                    X_val_fold = self.X[val_idx]
                                    y_train_fold = self.y[train_idx]
                                    y_val_fold = self.y[val_idx]
                                    
                                    # Create model with current hyperparameters
                                    model = MultiLayerSCTNClassifier(
                                        input_size=self.X.shape[1],
                                        hidden_layers=hidden_layers,
                                        learning_rate=learning_rate,
                                        signal_type=self.signal_type,
                                        threshold_range=threshold_range,
                                        weight_init_std=weight_std
                                    )
                                    
                                    # Train model
                                    model.train(X_train_fold, y_train_fold, epochs=epochs, verbose=False)
                                    
                                    # Evaluate
                                    y_pred_fold = model.predict(X_val_fold)
                                    fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
                                    cv_accuracies.append(fold_accuracy)
                                
                                mean_accuracy = np.mean(cv_accuracies)
                                std_accuracy = np.std(cv_accuracies)
                                
                                # Store result
                                config = {
                                    'hidden_layers': hidden_layers,
                                    'learning_rate': learning_rate,
                                    'weight_init_std': weight_std,
                                    'threshold_range': threshold_range,
                                    'epochs': epochs,
                                    'mean_accuracy': mean_accuracy,
                                    'std_accuracy': std_accuracy,
                                    'architecture': f"{self.X.shape[1]}-{'-'.join(map(str, hidden_layers))}-1"
                                }
                                
                                self.results.append(config)
                                
                                print(f"          ✅ Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
                                
                                # Check if this is the best so far
                                if mean_accuracy > best_accuracy:
                                    best_accuracy = mean_accuracy
                                    best_config = config
                                    print(f"          🏆 NEW BEST! {best_accuracy:.4f}")
                                
                            except Exception as e:
                                print(f"          ❌ Failed: {str(e)}")
                                continue
        
        print(f"\n🎉 NESTED SEARCH COMPLETED!")
        print(f"   Tested {len(self.results)} successful combinations")
        print(f"   Best accuracy: {best_accuracy:.4f}")
        
        return best_config, self.results
    
    def analyze_results(self, top_n=10):
        """Analyze and display the best results"""
        if not self.results:
            print("No results to analyze!")
            return
        
        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x['mean_accuracy'], reverse=True)
        
        print(f"\n📊 TOP {min(top_n, len(sorted_results))} CONFIGURATIONS")
        print("=" * 120)
        print(f"{'Rank':<5} {'Architecture':<25} {'Accuracy':<12} {'±Std':<8} {'LR':<6} {'WStd':<6} {'TRange':<12} {'Epochs':<7}")
        print("-" * 120)
        
        for i, result in enumerate(sorted_results[:top_n]):
            rank = i + 1
            arch = result['architecture']
            acc = result['mean_accuracy']
            std = result['std_accuracy']
            lr = result['learning_rate']
            wstd = result['weight_init_std']
            trange = f"{result['threshold_range'][0]:.0f}-{result['threshold_range'][1]:.0f}"
            epochs = result['epochs']
            
            print(f"{rank:<5} {arch:<25} {acc:.4f}      {std:.4f}   {lr:<6} {wstd:<6} {trange:<12} {epochs:<7}")
        
        # Best configuration analysis
        best = sorted_results[0]
        print(f"\n🏆 BEST CONFIGURATION DETAILS:")
        print("=" * 50)
        print(f"Accuracy: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")
        print(f"Architecture: {best['architecture']}")
        print(f"Hidden Layers: {best['hidden_layers']}")
        print(f"Learning Rate: {best['learning_rate']}")
        print(f"Weight Init Std: {best['weight_init_std']}")
        print(f"Threshold Range: {best['threshold_range']}")
        print(f"Epochs: {best['epochs']}")
        
        return sorted_results

def run_architecture_optimization():
    """Main function to run the complete architecture optimization"""
    print("🚀 MULTI-LAYER SCTN ARCHITECTURE OPTIMIZATION")
    print("=" * 80)
    print("Testing multiple architectures with nested hyperparameter loops")
    print("to find configurations that improve accuracy over single neuron")
    print()
    
    # Load data
    extractor = ProductionFeatureExtractor(chunk_dir=CHUNKED_OUTPUT_DIR)
    datasets = extractor.load_production_datasets()
    
    if not datasets:
        print("❌ No datasets loaded! Make sure chunked data exists.")
        return None
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, (X, y) in datasets.items():
        print(f"\n🎯 OPTIMIZING {dataset_name.upper()} CLASSIFIER")
        print("=" * 60)
        
        # Create tuner
        tuner = HyperparameterTuner(X, y, signal_type=dataset_name)
        
        # Run nested grid search
        best_config, all_configs = tuner.run_nested_grid_search()
        
        # Analyze results
        top_configs = tuner.analyze_results(top_n=5)
        
        # Store results
        all_results[dataset_name] = {
            'best_config': best_config,
            'all_configs': all_configs,
            'top_configs': top_configs
        }
        
        # Train and evaluate final model
        if best_config:
            print(f"\n🏁 TRAINING FINAL MODEL WITH BEST ARCHITECTURE")
            print("-" * 50)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )
            
            # Create best model
            final_model = MultiLayerSCTNClassifier(
                input_size=X.shape[1],
                hidden_layers=best_config['hidden_layers'],
                learning_rate=best_config['learning_rate'],
                signal_type=dataset_name,
                threshold_range=best_config['threshold_range'],
                weight_init_std=best_config['weight_init_std']
            )
            
            # Train final model
            final_model.train(X_train, y_train, epochs=best_config['epochs'])
            
            # Final evaluation
            y_pred = final_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n🎯 FINAL TEST RESULTS:")
            print(f"   Architecture: {best_config['architecture']}")
            print(f"   CV Accuracy: {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}")
            print(f"   Final Test Accuracy: {final_accuracy:.4f}")
            
            improvement = final_accuracy - best_config['mean_accuracy']
            print(f"   Improvement over CV: {improvement:+.4f}")
            
            # Classification report
            print(f"\n📋 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Noise', 'Signal']))
            
            # Save best model
            model_path = f"optimized_multilayer_{dataset_name}_sctn.pkl"
            model_data = {
                'model': final_model,
                'config': best_config,
                'final_accuracy': final_accuracy,
                'architecture': best_config['architecture'],
                'improvement': improvement
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Optimized model saved: {model_path}")
            
            all_results[dataset_name]['final_model'] = final_model
            all_results[dataset_name]['final_accuracy'] = final_accuracy
            all_results[dataset_name]['improvement'] = improvement
    
    # Final summary
    print(f"\n🏆 OPTIMIZATION COMPLETE - SUMMARY")
    print("=" * 80)
    
    for dataset_name, results in all_results.items():
        if 'final_accuracy' in results:
            print(f"\n{dataset_name.upper()} RESULTS:")
            print(f"  Best Architecture: {results['best_config']['architecture']}")
            print(f"  CV Accuracy: {results['best_config']['mean_accuracy']:.4f}")
            print(f"  Final Test Accuracy: {results['final_accuracy']:.4f}")
            print(f"  Improvement: {results['improvement']:+.4f}")
            
            if results['improvement'] > 0:
                print(f"  🎉 IMPROVED OVER BASELINE!")
            else:
                print(f"  📊 No improvement over baseline")
    
    return all_results

if __name__ == "__main__":
    results = run_architecture_optimization() 