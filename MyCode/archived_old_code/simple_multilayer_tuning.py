#!/usr/bin/env python3

# Simple Multi-Layer SCTN Architecture Optimization
import numpy as np
import time
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import SCTN components
import sys
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY
from resonator_work import ProductionFeatureExtractor, CHUNKED_OUTPUT_DIR, SCTNClassifier

def test_multilayer_architecture(X_train, y_train, X_test, y_test, 
                                hidden_layers, learning_rate, epochs):
    """Test a multi-layer architecture"""
    
    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create layers of SCTN neurons
    layers = []
    layer_sizes = [X_train.shape[1]] + hidden_layers + [1]
    
    # Create each layer
    for layer_idx in range(1, len(layer_sizes)):
        prev_size = layer_sizes[layer_idx - 1]
        current_size = layer_sizes[layer_idx]
        layer_neurons = []
        
        for neuron_idx in range(current_size):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.normal(0, 0.1, prev_size).astype(np.float64)
            neuron.threshold_pulse = np.random.uniform(20.0, 60.0)
            neuron.activation_function = BINARY
            neuron.theta = 0.0
            neuron.reset_to = 0.0
            neuron.membrane_should_reset = True
            layer_neurons.append(neuron)
        
        layers.append(layer_neurons)
    
    # Simple training loop
    for epoch in range(epochs):
        for i in range(len(X_train_scaled)):
            features = X_train_scaled[i]
            target = y_train[i]
            
            # Forward pass
            current_activations = features
            
            for layer in layers:
                next_activations = []
                for neuron in layer:
                    neuron.membrane_potential = 0.0
                    if len(current_activations) == len(neuron.synapses_weights):
                        activation = np.dot(current_activations, neuron.synapses_weights)
                    else:
                        min_size = min(len(current_activations), len(neuron.synapses_weights))
                        activation = np.dot(current_activations[:min_size], neuron.synapses_weights[:min_size])
                    
                    neuron.membrane_potential = activation
                    output = neuron._activation_function_binary()
                    next_activations.append(output)
                
                current_activations = np.array(next_activations)
            
            # Get prediction
            prediction = current_activations[0] if len(current_activations) > 0 else 0
            
            # Simple weight update for output layer only
            error = target - prediction
            if len(layers) > 0:
                output_layer = layers[-1]
                
                if len(layers) > 1:
                    prev_activations = []
                    for neuron in layers[-2]:
                        prev_activations.append(neuron.membrane_potential)
                    prev_activations = np.array(prev_activations)
                else:
                    prev_activations = features
                
                for neuron in output_layer:
                    if len(prev_activations) == len(neuron.synapses_weights):
                        neuron.synapses_weights += learning_rate * error * prev_activations
                    neuron.threshold_pulse += learning_rate * error * 0.1
                    neuron.threshold_pulse = np.clip(neuron.threshold_pulse, 15.0, 75.0)
    
    # Test the trained network
    correct = 0
    for i in range(len(X_test_scaled)):
        features = X_test_scaled[i]
        target = y_test[i]
        
        # Forward pass
        current_activations = features
        
        for layer in layers:
            next_activations = []
            for neuron in layer:
                neuron.membrane_potential = 0.0
                if len(current_activations) == len(neuron.synapses_weights):
                    activation = np.dot(current_activations, neuron.synapses_weights)
                else:
                    min_size = min(len(current_activations), len(neuron.synapses_weights))
                    activation = np.dot(current_activations[:min_size], neuron.synapses_weights[:min_size])
                
                neuron.membrane_potential = activation
                output = neuron._activation_function_binary()
                next_activations.append(output)
            
            current_activations = np.array(next_activations)
        
        prediction = current_activations[0] if len(current_activations) > 0 else 0
        
        if prediction == target:
            correct += 1
    
    accuracy = correct / len(X_test_scaled)
    return accuracy

def main():
    print("🚀 MULTI-LAYER SCTN ARCHITECTURE OPTIMIZATION")
    print("=" * 60)
    print("Testing different architectures with nested loops")
    print()
    
    # Load data
    extractor = ProductionFeatureExtractor(chunk_dir=CHUNKED_OUTPUT_DIR)
    datasets = extractor.load_production_datasets()
    
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    all_results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n🎯 Testing {dataset_name.upper()} architectures")
        print("-" * 40)
        
        # Define architectures to test
        architectures = [
            [],           # Single neuron (baseline)
            [8],          # Small hidden layer
            [12],
            [16],         # Medium hidden layer
            [20],
            [24],         # Large hidden layer
            [12, 6],      # Two layers decreasing
            [16, 8],
            [20, 10],
            [24, 12],
            [16, 12],     # Two layers mixed
            [20, 15],
            [16, 12, 6],  # Three layers
            [20, 15, 10],
            [24, 18, 12],
        ]
        
        learning_rates = [0.05, 0.1, 0.15, 0.2]
        epochs_list = [50, 75, 100]
        
        best_accuracy = 0
        best_config = None
        
        total_combos = len(architectures) * len(learning_rates) * len(epochs_list)
        print(f"Testing {total_combos} combinations...")
        print()
        
        combo_count = 0
        
        # NESTED LOOPS FOR COMPREHENSIVE TESTING
        for arch_idx, hidden_layers in enumerate(architectures):
            arch_str = f"{X.shape[1]}-{'-'.join(map(str, hidden_layers)) if hidden_layers else 'single'}-1"
            total_neurons = sum(hidden_layers) + 1 if hidden_layers else 1
            
            print(f"🏗️  Architecture {arch_idx + 1}/{len(architectures)}: {arch_str} ({total_neurons} neurons)")
            
            for lr_idx, lr in enumerate(learning_rates):
                print(f"  📈 LR {lr_idx + 1}/{len(learning_rates)}: {lr}")
                
                for ep_idx, epochs in enumerate(epochs_list):
                    combo_count += 1
                    print(f"    ⏱️  Epochs {ep_idx + 1}/{len(epochs_list)}: {epochs} [{combo_count}/{total_combos}]", end=" ")
                    
                    try:
                        # Split data for testing
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, stratify=y, random_state=42
                        )
                        
                        # Test configuration
                        if len(hidden_layers) == 0:
                            # Single neuron baseline
                            model = SCTNClassifier(
                                input_size=X.shape[1],
                                signal_type=dataset_name
                            )
                            model.train(X_train, y_train, epochs=epochs, lr=lr, verbose=False)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                        else:
                            # Multi-layer network
                            accuracy = test_multilayer_architecture(
                                X_train, y_train, X_test, y_test,
                                hidden_layers, lr, epochs
                            )
                        
                        print(f"→ {accuracy:.4f}")
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_config = {
                                'architecture': arch_str,
                                'hidden_layers': hidden_layers,
                                'learning_rate': lr,
                                'epochs': epochs,
                                'accuracy': accuracy,
                                'total_neurons': total_neurons,
                                'is_multilayer': len(hidden_layers) > 0
                            }
                            print(f"      🏆 NEW BEST! {best_accuracy:.4f}")
                        
                    except Exception as e:
                        print(f"→ Failed: {str(e)[:30]}")
                        continue
        
        # Store results
        all_results[dataset_name] = best_config
        
        # Display results for this dataset
        print(f"\n🏆 BEST CONFIGURATION FOR {dataset_name.upper()}:")
        if best_config:
            print(f"   Architecture: {best_config['architecture']}")
            print(f"   Hidden Layers: {best_config['hidden_layers']}")
            print(f"   Total Neurons: {best_config['total_neurons']}")
            print(f"   Learning Rate: {best_config['learning_rate']}")
            print(f"   Epochs: {best_config['epochs']}")
            print(f"   Accuracy: {best_config['accuracy']:.4f}")
            
            if best_config['is_multilayer']:
                print(f"   🎉 MULTI-LAYER NETWORK WINS!")
            else:
                print(f"   📊 Single neuron remains optimal")
            
            # Save configuration
            with open(f"best_architecture_{dataset_name}.pkl", 'wb') as f:
                pickle.dump(best_config, f)
            print(f"   💾 Saved: best_architecture_{dataset_name}.pkl")
        else:
            print("   No successful configurations found")
    
    # Final summary
    print(f"\n🏆 FINAL OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for dataset_name, config in all_results.items():
        if config:
            print(f"\n{dataset_name.upper()}:")
            print(f"  Best: {config['architecture']} → {config['accuracy']:.4f}")
            if config['is_multilayer']:
                print(f"  ✅ Multi-layer improvement achieved!")
            else:
                print(f"  📊 Single neuron optimal")
    
    return all_results

if __name__ == "__main__":
    results = main()
