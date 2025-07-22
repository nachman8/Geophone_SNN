import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import gc

# Set up proper path for snn module
project_root = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, project_root)

# Import the resonator functions
from sctnN.resonator import simple_resonator, test_resonator_on_chirp, create_chirp_signal
from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator
from sctnN.spiking_network import SpikingNetwork
from sctnN.spiking_neuron import create_SCTN
from sctnN.layers import SCTNLayer

def custom_resonator_output_spikes(
        resonator,
        freq0,
        clk_freq,
        sample_rate=1,
        step=None,
        save_figure=False,
        path=None,
        plot=True,
        c=None
):
    resonator.log_out_spikes(-1)
    # Test ALL resonators over the same frequency range to see frequency selectivity
    start_freq = 15  # Fixed start frequency for all resonators
    end_freq = 105   # Fixed end frequency for all resonators  
    spectrum = end_freq - start_freq
    step = step or 1 / clk_freq / sample_rate
    test_size = int(spectrum / step)
    
    # Use the existing test_resonator_on_chirp function
    test_resonator_on_chirp(
        resonator,
        test_size=test_size,
        start_freq=start_freq,
        step=step,
        clk_freq=clk_freq,
        amplifier=1
    )

    spikes_neuron = resonator.neurons[-1]
    events = spikes_neuron.out_spikes()
    
    if len(events) == 0:
        # If no spikes, create empty array with proper length
        y_spikes = np.zeros(test_size + 1000)  # Add buffer
    else:
        y_spikes = np.zeros(max(events[-1] + 1, test_size + 1000))
        y_spikes[events] = 1

    if path is not None:
        np.savez_compressed(path, spikes=y_spikes)

    spikes_window_size = 500
    y_spikes = np.convolve(y_spikes, np.ones(spikes_window_size, dtype=int), 'valid')
    x = np.linspace(start_freq, end_freq, len(y_spikes))
    
    plt.plot(x, y_spikes, label=freq0, c=c)
    plt.fill_between(x, 0, y_spikes, alpha=0.3, color=c)
    
    if save_figure:
        plt.savefig(f'resonator_f_{freq0:.3f}.png', bbox_inches='tight')
        plt.close()
    elif plot:
        plt.show()
    
    return x, y_spikes

def main():
    print("Starting sequential resonator processing...")
    
    # Show available resonators first
    print("Available resonator frequencies:")
    print(sorted(RESONATOR_FUNCTIONS.keys()))
    print(f"Total available resonators: {len(RESONATOR_FUNCTIONS)}")
    
    # Set up the plot
    plt.figure(figsize=(20, 5))
    
    # Your specific frequencies organized by bands
    resonators_freqs = [
        # LOW_FREQ (20-30 Hz)
        [22.1, 28.8],
        # CAR_APPROACH + CAR_PEAK (30-40 Hz)
        [30.5, 33.9, 34.7, 37.2],
        # CAR_TAIL + MID_GAP (40-60 Hz) 
        [40.2, 41.2, 43.6, 47.7, 50.9, 52.6, 58.7],
        # HUMAN_PEAK + HUMAN_TAIL (60-80 Hz)
        [63.6, 69.4, 76.3],
        # HIGH_FREQ (85-100 Hz)
        [89.8, 95.4]
    ]
    
    total_resonators = sum(len(freqs) for freqs in resonators_freqs)
    print(f"Processing {total_resonators} resonators sequentially...")
    
    processed_count = 0
    
    for i, freqs in enumerate(resonators_freqs):
        for j, freq0 in enumerate(freqs):
            print(f"Processing resonator {processed_count + 1}/{total_resonators}: freq0={freq0}")
            
            try:
                # Determine clock frequency based on reference code
                if freq0 < 10:
                    clk_freq = 15360
                else:
                    clk_freq = 153600

                # Set up colormap for this band (same as reference)
                c = np.arange(0, len(freqs) + 4)
                norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
                
                # Determine step size and colormap based on band
                if i == 0:  # LOW_FREQ
                    step = 1/15360/10  # ≈ 0.00651 Hz
                    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Blues'))
                elif i == 1:  # CAR_APPROACH + CAR_PEAK
                    step = 1/15360/10  # ≈ 0.00651 Hz
                    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Purples'))
                elif i == 2:  # CAR_TAIL + MID_GAP
                    step = 1/15360/8   # ≈ 0.00814 Hz
                    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Greens'))
                elif i == 3:  # HUMAN_PEAK + HUMAN_TAIL
                    step = 1/15360/6   # ≈ 0.01085 Hz
                    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Oranges'))
                else:  # HIGH_FREQ
                    step = 1/15360/5   # ≈ 0.01302 Hz
                    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Reds'))

                cmap.set_array([])
                color = cmap.to_rgba(j + 4)
                
                # Create resonator using the available functions
                try:
                    resonator_func, actual_freq = get_closest_resonator(freq0)
                    resonator = resonator_func()
                    print(f"Using closest resonator: {actual_freq} for target {freq0}")
                except Exception as e:
                    print(f"Error creating resonator for {freq0}: {e}")
                    continue
                
                # Process the resonator
                x, y_spikes = custom_resonator_output_spikes(
                    resonator, 
                    freq0, 
                    clk_freq, 
                    step=step, 
                    plot=False,  
                    c=color
                )
                
                # Clean up memory
                del resonator
                gc.collect()
                
                processed_count += 1
                print(f"Successfully processed resonator {freq0}")
                
            except Exception as e:
                print(f"Error processing resonator {freq0}: {e}")
                continue

    print(f"Successfully processed {processed_count} resonators")
    
    # Plot formatting - adapted for your frequency range
    plt.xlim([15, 105])
    plt.ylim([0, 500])
    plt.yticks(fontsize=13)
    plt.xticks([20, 30, 40, 48, 60, 80, 90, 100], fontsize=13)
    
    # Band labels using your proper band names
    plt.text(25, 470, 'LOW_FREQ', fontsize=14, color='k', weight='bold')
    plt.text(35, 470, 'CAR_APPROACH', fontsize=14, color='k', weight='bold')
    plt.text(50, 470, 'CAR_TAIL+MID_GAP', fontsize=14, color='k', weight='bold')
    plt.text(70, 470, 'HUMAN_PEAK', fontsize=14, color='k', weight='bold')
    plt.text(92, 470, 'HIGH_FREQ', fontsize=14, color='k', weight='bold')

    # Vertical lines to separate bands
    plt.vlines(20, 0, 450, colors='k', linestyles='--')
    plt.vlines(30, 0, 450, colors='k', linestyles='--')
    plt.vlines(40, 0, 450, colors='k', linestyles='--')
    plt.vlines(60, 0, 450, colors='k', linestyles='--')
    plt.vlines(80, 0, 450, colors='k', linestyles='--')
    plt.vlines(100, 0, 450, colors='k', linestyles='--')
    
    plt.ylabel('Spikes Rate', fontsize=20)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    
    # Save plot
    plt.savefig('geophone_resonators_array_sequential.pdf')
    plt.show()
    
    print("Sequential processing completed successfully!")

if __name__ == "__main__":
    main()