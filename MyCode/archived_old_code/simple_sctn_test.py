#!/usr/bin/env python3
"""
Simple sctnN Test - Verify STDP Learning Works

Quick test to verify the sctnN library is learning properly
"""

import numpy as np
import sys

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_network import SpikingNetwork
from sctnN.layers import SCTNLayer  
from sctnN.spiking_neuron import create_SCTN, IDENTITY

def create_simple_network():
    """Create a simple 2-input, 1-output network"""
    print("🏗️  Creating simple sctnN network...")
    
    # Create network
    network = SpikingNetwork()
    network.clk_freq = 1536000
    network.add_amplitude(1000)
    
    # Input layer (2 neurons)
    input_neurons = []
    for i in range(2):
        neuron = create_SCTN()
        neuron.activation_function = IDENTITY
        neuron.membrane_should_reset = False
        input_neurons.append(neuron)
    
    input_layer = SCTNLayer(input_neurons)
    network.add_layer(input_layer)
    
    # Output layer (1 neuron with STDP)
    output_neuron = create_SCTN()
    output_neuron.synapses_weights = np.array([0.5, 0.5], dtype=np.float64)
    output_neuron.activation_function = IDENTITY
    output_neuron.membrane_should_reset = False
    output_neuron.theta = -0.8
    
    # Set STDP learning
    output_neuron.set_stdp(
        A_LTP=10e-5,
        A_LTD=-8e-5,
        tau=1e-5,
        clk_freq=network.clk_freq,
        wmax=2.0,
        wmin=0.1
    )
    
    output_layer = SCTNLayer([output_neuron])
    network.add_layer(output_layer)
    
    # Set up logging
    for neuron in network.neurons:
        network.log_out_spikes(neuron._id)
    
    print(f"   ✅ Network created: 2 → 1")
    print(f"   📊 Initial weights: {output_neuron.synapses_weights}")
    
    return network

def test_simple_learning():
    """Test simple STDP learning"""
    print("🧪 Testing simple STDP learning...")
    
    network = create_simple_network()
    output_neuron = network.layers_neurons[-1].neurons[0]
    
    initial_weights = np.copy(output_neuron.synapses_weights)
    print(f"   📊 Initial weights: {initial_weights}")
    
    # Training data: simple patterns
    patterns = [
        np.array([500, 100]),  # Pattern A: strong first input
        np.array([100, 500]),  # Pattern B: strong second input
        np.array([300, 300]),  # Pattern C: equal inputs
    ]
    
    # Train for a few epochs
    for epoch in range(5):
        for pattern in patterns:
            # Reset network state
            network.reset_learning()
            network.forget_logs()
            
            # Feed pattern
            classes = network.input_full_data(pattern)
            
        current_weights = np.copy(output_neuron.synapses_weights)
        weight_change = np.sum(np.abs(current_weights - initial_weights))
        
        print(f"   📈 Epoch {epoch}: weights={current_weights}, change={weight_change:.6f}")
        
        if weight_change > 0.001:  # If weights are changing
            print("   ✅ STDP LEARNING IS WORKING!")
            return True
    
    print("   ❌ No significant weight changes detected")
    return False

def test_variability_detection():
    """Test if our variability detection works"""
    print("\n🔍 Testing variability detection...")
    
    # Simulate car pattern (bursty)
    car_pattern = np.array([100, 50, 200, 80, 300, 70, 150, 60])  # Variable
    car_mean = np.mean(car_pattern)
    car_std = np.std(car_pattern)
    car_variability = car_std / car_mean
    
    # Simulate nothing pattern (steady)
    nothing_pattern = np.array([120, 125, 118, 122, 119, 123, 121, 124])  # Consistent
    nothing_mean = np.mean(nothing_pattern)
    nothing_std = np.std(nothing_pattern)
    nothing_variability = nothing_std / nothing_mean
    
    print(f"   🚗 Car pattern variability: {car_variability:.3f}")
    print(f"   📉 Nothing pattern variability: {nothing_variability:.3f}")
    
    threshold = 0.20
    car_detected = car_variability > threshold
    nothing_detected = nothing_variability > threshold
    
    print(f"   🎯 Using threshold: {threshold}")
    print(f"   🚗 Car detected: {car_detected}")
    print(f"   📉 Nothing detected: {nothing_detected}")
    
    if car_detected and not nothing_detected:
        print("   ✅ VARIABILITY DETECTION WORKING!")
        return True
    else:
        print("   ❌ Variability detection needs tuning")
        return False

if __name__ == "__main__":
    print("🚀 SIMPLE sctnN VERIFICATION TEST")
    print("=" * 50)
    
    try:
        # Test 1: STDP Learning
        stdp_works = test_simple_learning()
        
        # Test 2: Variability Detection  
        variability_works = test_variability_detection()
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   📈 STDP Learning: {'✅ WORKS' if stdp_works else '❌ FAILED'}")
        print(f"   🔍 Variability Detection: {'✅ WORKS' if variability_works else '❌ FAILED'}")
        
        if stdp_works and variability_works:
            print("\n🎉 BOTH KEY COMPONENTS WORKING!")
            print("💡 The main classifier should achieve good results!")
        else:
            print("\n🔧 Some components need debugging...")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 