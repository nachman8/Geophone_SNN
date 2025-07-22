#!/usr/bin/env python3
"""
Examine Raw Resonator Spike Events
Understand the format and content of raw spike data from chunks
"""

import numpy as np
import pickle
import os

def examine_chunk_spikes():
    """Examine what's actually in the raw resonator spike data"""
    
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    
    print("🔍 EXAMINING RAW RESONATOR SPIKE EVENTS")
    print("=" * 60)
    
    # Look at car chunk
    car_chunk_dir = os.path.join(chunks_dir, "car")
    index_file = os.path.join(car_chunk_dir, "chunk_index.pkl")
    
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        # Load first chunk
        first_chunk_file = chunk_index['chunk_files'][0]
        print(f"📁 Loading first car chunk: {first_chunk_file}")
        
        with open(first_chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
        
        print(f"\n📊 CHUNK DATA STRUCTURE:")
        for key in chunk_data.keys():
            print(f"   {key}: {type(chunk_data[key])}")
        
        print(f"\n🎯 RESONATOR OUTPUTS:")
        resonator_outputs = chunk_data['resonator_outputs']
        print(f"Type: {type(resonator_outputs)}")
        print(f"Keys: {list(resonator_outputs.keys())}")
        
        for clk_freq, spikes_arrays in resonator_outputs.items():
            print(f"\n⏰ Clock frequency: {clk_freq}")
            print(f"   Number of resonators: {len(spikes_arrays)}")
            
            for i, spike_events in enumerate(spikes_arrays[:3]):  # First 3 resonators
                print(f"   Resonator {i}:")
                print(f"      Type: {type(spike_events)}")
                print(f"      Shape/Length: {len(spike_events) if hasattr(spike_events, '__len__') else 'N/A'}")
                if len(spike_events) > 0:
                    print(f"      First 10 spikes: {spike_events[:10]}")
                    print(f"      Min: {np.min(spike_events)}, Max: {np.max(spike_events)}")
                    print(f"      Sample rate check: {spike_events[-1] / clk_freq:.2f}s duration")
                else:
                    print(f"      No spikes detected")
        
        print(f"\n📈 PROCESSED SPECTROGRAM (for comparison):")
        spikes_bands_spectrogram = chunk_data['spikes_bands_spectrogram']
        print(f"Shape: {spikes_bands_spectrogram.shape}")
        print(f"Type: {type(spikes_bands_spectrogram)}")
        print(f"Value range: {np.min(spikes_bands_spectrogram):.3f} to {np.max(spikes_bands_spectrogram):.3f}")
        print(f"First band stats: mean={np.mean(spikes_bands_spectrogram[0]):.3f}, std={np.std(spikes_bands_spectrogram[0]):.3f}")
        
        # Compare car vs car_nothing
        print(f"\n🚗 vs 🚫 COMPARISON:")
        
        # Load car_nothing chunk
        nothing_chunk_dir = os.path.join(chunks_dir, "car_nothing")
        nothing_index_file = os.path.join(nothing_chunk_dir, "chunk_index.pkl")
        
        if os.path.exists(nothing_index_file):
            with open(nothing_index_file, 'rb') as f:
                nothing_index = pickle.load(f)
            
            nothing_chunk_file = nothing_index['chunk_files'][0]
            with open(nothing_chunk_file, 'rb') as f:
                nothing_data = pickle.load(f)
            
            car_resonator_outputs = chunk_data['resonator_outputs']
            nothing_resonator_outputs = nothing_data['resonator_outputs']
            
            print(f"Car chunk spikes vs Nothing chunk spikes:")
            
            for clk_freq in car_resonator_outputs:
                car_spikes = car_resonator_outputs[clk_freq]
                nothing_spikes = nothing_resonator_outputs[clk_freq]
                
                print(f"\n  Clock {clk_freq}:")
                for i, (car_events, nothing_events) in enumerate(zip(car_spikes[:5], nothing_spikes[:5])):
                    car_count = len(car_events)
                    nothing_count = len(nothing_events)
                    ratio = car_count / (nothing_count + 1e-10)
                    
                    print(f"    Resonator {i}: Car={car_count} spikes, Nothing={nothing_count} spikes, Ratio={ratio:.2f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   - Raw spike events are arrays of spike timestamps")
        print(f"   - Need to convert these to features or spike trains for sctnN")
        print(f"   - Current feature extraction might be losing temporal information")
        print(f"   - Consider using spike trains directly instead of statistical features")

if __name__ == "__main__":
    examine_chunk_spikes() 