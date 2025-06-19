import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import re
import sys
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"  # Parent directory of sctnN
sys.path.insert(0, sctn_parent_dir)
from sctnN.resonator import simple_resonator, test_resonator_on_chirp

def create_chirp_signal(clk_freq, start_freq, end_freq):
    test_size = clk_freq * (end_freq - start_freq)
    step = 1/clk_freq
    sine_wave = (np.arange(test_size) * step + start_freq + step)
    sine_wave = sine_wave * 2 * np.pi / clk_freq
    sine_wave = np.cumsum(sine_wave)
    return np.sin(sine_wave)

def plot_emitted_spikes(network, x_stop, nid=-1, label=None, spare_steps=10, ax=None):
    """Plot spikes on a specific axis (or current axis if None) and return frequency with max spike"""
    if ax is None:
        ax = plt.gca()  # Get current axis if none provided
        
    spikes_neuron = network.neurons[nid]
    y_events = spikes_neuron.out_spikes()
    if len(y_events) == 0:
        return None, None
    
    y_spikes = np.zeros(y_events[-1] + 1)
    y_spikes[y_events] = 1
    y_spikes = np.convolve(y_spikes, np.ones(500, dtype=int), 'valid')
    y_spikes = y_spikes[::spare_steps]
    x = np.linspace(0, x_stop, len(y_spikes))
    
    # Find the frequency with maximum spike
    max_spike_idx = np.argmax(y_spikes)
    max_spike_freq = x[max_spike_idx]
    max_spike_value = y_spikes[max_spike_idx]
    
    # Plot the line
    ax.plot(x, y_spikes, label=label)
    
    # Highlight the maximum spike
    ax.scatter(max_spike_freq, max_spike_value, color='red', s=100, zorder=3)
    ax.axvline(x=max_spike_freq, color='r', linestyle='--', alpha=0.5)
    
    return max_spike_freq, max_spike_value
    
def parse_resonator_data(text):
    """Parse the resonator parameter data from the text file"""
    resonator_data = []
    
    # Pattern to match a complete resonator data block
    pattern = r"File: (f_[\d\.]+\.json)\s+freq0: ([\d\.]+)\s+clk_freq: (\d+)\s+lf: (\d+)\s+weight_results: \[([\d\.\s,]+)\]\s+theta_results: \[([\-\d\.\s,]+)\]"
    
    matches = re.finditer(pattern, text)
    
    for match in matches:
        filename = match.group(1)
        freq0 = float(match.group(2))
        clk_freq = int(match.group(3))
        lf = int(match.group(4))
        
        # Parse weight and theta results
        weight_str = match.group(5)
        weights = [float(w.strip()) for w in weight_str.split(',')]
        
        theta_str = match.group(6)
        thetas = [float(t.strip()) for t in theta_str.split(',')]
        
        resonator_data.append({
            'filename': filename,
            'freq0': freq0,
            'clk_freq': clk_freq,
            'lf': lf,
            'weights': weights,
            'thetas': thetas
        })
    
    return resonator_data

def process_resonator(params):
    """Process a single resonator with the given parameters"""
    freq0 = params['freq0']
    clk_freq = params['clk_freq']
    lf = params['lf']
    thetas = params['thetas']
    weights = params['weights']
    
    # Create the resonator
    resonator = simple_resonator(
        freq0=freq0,
        clk_freq=clk_freq,
        lf=lf,
        thetas=thetas,
        weights=weights,
    )
    
    # Log the spikes emitted from the last neuron
    resonator.log_out_spikes(-1)
    
    # Define the chirp signal parameters - focus on 0-150 Hz range
    spectrum = 150  # Changed from 35 to 150
    step = 1/20000
    test_size = int(spectrum / step)
    
    # Test the resonator with the chirp signal
    test_resonator_on_chirp(
        resonator,
        start_freq=0,
        test_size=test_size,
        clk_freq=clk_freq,
        step=step
    )
    
    return {
        'freq0': freq0,
        'resonator': resonator
    }

def main():
    # Read the text file with resonator parameters
    with open('/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/examine/txt/resonator2.txt', 'r') as f:
        text = f.read()
    
    # Parse the resonator data from the text
    resonator_data = parse_resonator_data(text)
    
    # Process the resonators in parallel using joblib with 15 CPUs
    results = Parallel(n_jobs=15, verbose=1)(
        delayed(process_resonator)(params) for params in resonator_data
    )
    
    # Sort results by frequency for better visualization
    results.sort(key=lambda x: x['freq0'])
    
    # Create a summary of all resonators' max frequencies
    summary_data = []
    
    # Plot each resonator's response one by one
    for result in results:
        # Create a new figure for each resonator with a fresh axis
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot the spikes for this resonator on the specific axis and get max frequency
        max_freq, max_val = plot_emitted_spikes(
            result['resonator'], 
            x_stop=150,  # Changed from 35 to 150
            label=f"freq0={result['freq0']:.2f}",
            ax=ax  # Pass the specific axis to plot on
        )
        
        # Add to summary data
        summary_data.append({
            'designed_freq': result['freq0'],
            'max_response_freq': max_freq,
            'max_response_value': max_val
        })
        
        # Set title with the max frequency information
        ax.set_title(f'Resonator Output (designed for {result["freq0"]:.2f} Hz, max response at {max_freq:.2f} Hz)', 
                    fontsize=14)
        ax.set_ylabel('Spikes per Window (500)', fontsize=12)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        
        # Set x-axis range to 0-150 Hz
        ax.set_xlim(0, 150)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate the maximum spike
        ax.annotate(f'Max: {max_freq:.2f} Hz',
                   xy=(max_freq, max_val),
                   xytext=(max_freq+1, max_val),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=12)
        
        # Add legend, adjust layout, save this specific figure
        ax.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        # Save with a unique filename based on the resonator frequency
        filename = f'resonator_output_freq_{result["freq0"]:.2f}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Show the plot and wait for user to close it before proceeding to next one
        plt.show()
        
        # Explicitly close the figure to ensure it doesn't affect the next plot
        plt.close(fig)
    
    # Create a summary plot showing designed frequency vs actual max response frequency
    fig, ax = plt.subplots(figsize=(15, 10))
    
    designed_freqs = [data['designed_freq'] for data in summary_data]
    max_response_freqs = [data['max_response_freq'] for data in summary_data]
    
    # Plot the perfect line (y=x)
    max_freq = max(max(designed_freqs), max(max_response_freqs))
    ax.plot([0, max_freq], [0, max_freq], 'k--', label='Perfect match')
    
    # Plot the actual data
    ax.scatter(designed_freqs, max_response_freqs, s=100, alpha=0.7)
    
    # Add labels and annotations
    for i, data in enumerate(summary_data):
        ax.annotate(f"{data['designed_freq']:.2f}",
                   (data['designed_freq'], data['max_response_freq']),
                   xytext=(5, 5),
                   textcoords='offset points')
    
    ax.set_title('Designed Frequency vs Actual Max Response Frequency', fontsize=14)
    ax.set_xlabel('Designed Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Max Response Frequency (Hz)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('resonator_frequency_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print the summary table
    print("\nSummary of Resonator Frequencies:")
    print("---------------------------------")
    print(f"{'Designed (Hz)':<15} {'Max Response (Hz)':<20} {'Difference (Hz)':<15}")
    print("---------------------------------")
    for data in summary_data:
        diff = data['max_response_freq'] - data['designed_freq']
        print(f"{data['designed_freq']:<15.2f} {data['max_response_freq']:<20.2f} {diff:<15.2f}")

if __name__ == "__main__":
    main()