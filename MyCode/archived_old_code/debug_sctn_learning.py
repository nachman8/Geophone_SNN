#!/usr/bin/env python3
"""
Debug sctnN Learning
Simple test to verify if sctnN can learn with actual chunk data
"""

import numpy as np
import pickle
import os
import sys

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import create_SCTN
from sctnN.layers import SCTNLayer

def test_simple_learning():
    """Test if sctnN can learn simple patterns"""
    print("🔧 DEBUGGING sctnN LEARNING")
    print("=" * 50)
    
    # Create very simple network
    network = SpikingNetwork()
    network.clk_freq = 153600
    network.add_amplitude(1000)
    
    # Input layer (2 features)
    input_neurons = []
    for i in range(2):
        neuron = create_SCTN()
        neuron.activation_function = 0  # IDENTITY
        neuron.membrane_should_reset = False
        input_neurons.append(neuron)
    
    input_layer = SCTNLayer(input_neurons)
    network.add_layer(input_layer)
    
    # Output layer (1 neuron with STDP)
    output_neuron = create_SCTN()
    output_neuron.synapses_weights = np.array([0.5, 0.5], dtype=np.float64)
    output_neuron.activation_function = 0
    output_neuron.membrane_should_reset = False
    output_neuron.theta = -0.5
    
    # Set STDP
    output_neuron.set_stdp(
        A_LTP=10e-5, A_LTD=-8e-5, tau=1e-5,
        clk_freq=network.clk_freq, wmax=2.0, wmin=0.1
    )
    
    output_layer = SCTNLayer([output_neuron])
    network.add_layer(output_layer)
    
    # Enable logging
    for neuron in network.neurons:
        network.log_out_spikes(neuron._id)
    
    print(f"📊 Initial weights: {output_neuron.synapses_weights}")
    
    # Test with simple patterns
    print(f"\n🧪 Testing with simple patterns...")
    
    # Pattern A: [100, 0] -> should strengthen first weight
    # Pattern B: [0, 100] -> should strengthen second weight
    patterns = [
        ([100, 0], "Pattern A"),
        ([0, 100], "Pattern B"),
        ([100, 0], "Pattern A"),
        ([0, 100], "Pattern B"),
        ([100, 0], "Pattern A"),
    ]
    
    for i, (pattern, name) in enumerate(patterns):
        network.input_potential(np.array(pattern))
        
        # Check output
        output_spikes = len(output_neuron.out_spikes())
        
        print(f"  {name}: Input={pattern}, Output spikes={output_spikes}")
        print(f"    Weights: {output_neuron.synapses_weights}")
    
    print(f"\n📈 Final weights: {output_neuron.synapses_weights}")
    
    # Check if weights changed
    initial_weights = np.array([0.5, 0.5])
    final_weights = output_neuron.synapses_weights
    weight_change = np.abs(final_weights - initial_weights)
    
    print(f"📊 Weight changes: {weight_change}")
    
    if np.any(weight_change > 0.001):
        print("✅ LEARNING DETECTED! Weights changed significantly")
        return True
    else:
        print("❌ NO LEARNING! Weights unchanged")
        return False

def test_with_real_chunk_data():
    """Test learning with actual chunk data"""
    print(f"\n🗂️  TESTING WITH REAL CHUNK DATA")
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    # Load one car chunk and one nothing chunk
    car_chunk_file = None
    nothing_chunk_file = None
    
    # Find car chunk
    car_dir = os.path.join(chunks_dir, "car")
    car_index_file = os.path.join(car_dir, "chunk_index.pkl")
    if os.path.exists(car_index_file):
        with open(car_index_file, 'rb') as f:
            car_index = pickle.load(f)
        car_chunk_file = car_index['chunk_files'][0]
    
    # Find nothing chunk
    nothing_dir = os.path.join(chunks_dir, "car_nothing")
    nothing_index_file = os.path.join(nothing_dir, "chunk_index.pkl")
    if os.path.exists(nothing_index_file):
        with open(nothing_index_file, 'rb') as f:
            nothing_index = pickle.load(f)
        nothing_chunk_file = nothing_index['chunk_files'][0]
    
    if not car_chunk_file or not nothing_chunk_file:
        print("❌ Cannot find chunk files")
        return False
    
    # Load chunks
    with open(car_chunk_file, 'rb') as f:
        car_chunk = pickle.load(f)
    
    with open(nothing_chunk_file, 'rb') as f:
        nothing_chunk = pickle.load(f)
    
    print(f"📁 Loaded car chunk: {os.path.basename(car_chunk_file)}")
    print(f"📁 Loaded nothing chunk: {os.path.basename(nothing_chunk_file)}")
    
    # Extract simple features from processed spectrograms
    car_spikegram = car_chunk['spikes_bands_spectrogram']
    nothing_spikegram = nothing_chunk['spikes_bands_spectrogram']
    
    # Simple features: mean of each frequency band
    car_features = np.mean(car_spikegram, axis=1)
    nothing_features = np.mean(nothing_spikegram, axis=1)
    
    print(f"📊 Car features: {car_features[:4]} (mean={np.mean(car_features):.2f})")
    print(f"📊 Nothing features: {nothing_features[:4]} (mean={np.mean(nothing_features):.2f})")
    
    # Check if there's a clear difference
    difference = np.abs(car_features - nothing_features)
    max_diff = np.max(difference)
    print(f"📊 Max difference: {max_diff:.2f}")
    
    if max_diff < 1.0:
        print("⚠️  WARNING: Car and nothing features are very similar!")
        print("    This could explain why learning fails")
    
    # Create simple network for these features
    n_features = len(car_features)
    
    network = SpikingNetwork()
    network.clk_freq = 153600
    network.add_amplitude(100)
    
    # Input layer
    input_neurons = []
    for i in range(n_features):
        neuron = create_SCTN()
        neuron.activation_function = 0
        neuron.membrane_should_reset = False
        input_neurons.append(neuron)
    
    input_layer = SCTNLayer(input_neurons)
    network.add_layer(input_layer)
    
    # Output neuron
    output_neuron = create_SCTN()
    output_neuron.synapses_weights = np.random.normal(0.5, 0.1, n_features).astype(np.float64)
    output_neuron.synapses_weights = np.clip(output_neuron.synapses_weights, 0.1, 1.0)
    output_neuron.activation_function = 0
    output_neuron.membrane_should_reset = False
    output_neuron.theta = -0.5
    
    # Set STDP with higher learning rate
    output_neuron.set_stdp(
        A_LTP=50e-5,  # Higher learning rate
        A_LTD=-30e-5,
        tau=1e-5,
        clk_freq=network.clk_freq,
        wmax=2.0,
        wmin=0.1
    )
    
    output_layer = SCTNLayer([output_neuron])
    network.add_layer(output_layer)
    
    # Enable logging
    for neuron in network.neurons:
        network.log_out_spikes(neuron._id)
    
    initial_weights = output_neuron.synapses_weights.copy()
    print(f"📊 Initial weights (first 4): {initial_weights[:4]}")
    
    # Train alternating patterns
    print(f"\n🧠 Training with alternating car/nothing patterns...")
    
    for epoch in range(10):
        # Car pattern
        network.input_potential(car_features * 10)  # Scale up
        car_spikes = len(output_neuron.out_spikes())
        
        # Nothing pattern
        network.input_potential(nothing_features * 10)
        nothing_spikes = len(output_neuron.out_spikes())
        
        if epoch % 2 == 0:
            current_weights = output_neuron.synapses_weights
            weight_change = np.max(np.abs(current_weights - initial_weights))
            print(f"  Epoch {epoch+1}: Car={car_spikes}, Nothing={nothing_spikes}, Max weight change={weight_change:.6f}")
    
    final_weights = output_neuron.synapses_weights
    total_change = np.max(np.abs(final_weights - initial_weights))
    
    print(f"📊 Final weights (first 4): {final_weights[:4]}")
    print(f"📊 Total weight change: {total_change:.6f}")
    
    if total_change > 0.001:
        print("✅ LEARNING DETECTED with real chunk data!")
        return True
    else:
        print("❌ NO LEARNING with real chunk data")
        return False

def main():
    """Main debug function"""
    print("🚀 sctnN LEARNING DEBUG")
    print("Testing why sctnN isn't learning with our data")
    print()
    
    # Test 1: Simple patterns
    simple_works = test_simple_learning()
    
    # Test 2: Real chunk data
    real_works = test_with_real_chunk_data()
    
    print(f"\n🎯 DEBUG RESULTS:")
    print(f"   Simple patterns: {'✅ WORKS' if simple_works else '❌ FAILS'}")
    print(f"   Real chunk data: {'✅ WORKS' if real_works else '❌ FAILS'}")
    
    if simple_works and not real_works:
        print(f"\n💡 CONCLUSION: sctnN works, but real data lacks discriminative features")
        print(f"   🔧 Recommendation: Focus on better feature extraction")
    elif not simple_works:
        print(f"\n💡 CONCLUSION: sctnN setup issue - basic learning doesn't work")
        print(f"   🔧 Recommendation: Check sctnN parameters and setup")
    else:
        print(f"\n💡 CONCLUSION: Both work - issue might be elsewhere")

if __name__ == "__main__":
    main() 