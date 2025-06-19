import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import resample, spectrogram
from joblib import Parallel, delayed
import multiprocessing
import os
import time
import threading
from sklearn.model_selection import train_test_split


from pathlib import Path
DATA_DIR = Path.home() / "data"

# At the beginning of your resonator_work.py file
import sys
import os

# Add the directory CONTAINING sctnN to your Python path
# This should be the parent directory of sctnN
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"  # Parent directory of sctnN
sys.path.insert(0, sctn_parent_dir)

# Now you can import from sctnN
from sctnN.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator


import warnings
warnings.filterwarnings('ignore')


# Define separate resonator grids for different data types
clk_resonators_car = {
    153600: [
        # LOW_FREQ coverage for car
        10.5, 11.5, 12.8, 15.9, 16.6, 19.5, 22.1, 25.0, 26.8, 27.9, 28.8,
        # Enhanced CAR coverage (30-48 Hz) - all available for better car detection
        30.5, 34.7, 37.2, 40.2, 43.6,  47.7,
        # MID_GAP coverage
         52.6, 58.7,
        # Reduced HUMAN coverage - keep some for comparison
        63.6, 69.4, 76.3,
        # HIGH_FREQ coverage
        89.8
    ]
}

clk_resonators_human = {
    153600: [
        # Available LOW_FREQ coverage (20-30 Hz) - focus on human activity
        22.1,
        #26.8,
        #20.103782285292045,
        # Reduced CAR coverage (30-48 Hz) - keep minimal but essential
        #30.5, 33.9, 34.7, 40.2, 41.2, 47.7,
        # Enhanced MID_GAP coverage (48-60 Hz) - all available
        #50.9, 52.6,
        # ALL available HUMAN_PEAK and HUMAN_TAIL coverage (60-85 Hz)
        #63.6, 76.3,
        # Minimal HIGH_FREQ coverage (85-100 Hz)
        #95.4
    ],
    1536000: [
        # new resonator for better human detection
        # Available LOW_FREQ coverage (20-30 Hz) - focus on human activity
        #22.1, 25.0, 26.8,
        
        # Reduced CAR coverage (30-48 Hz) - keep minimal but essential
        #30.5, 33.9, 34.7, 40.2, 41.2, 47.7,
        33.72819986053411, 36.03508145476876, 40.52751866531022,
        # Enhanced MID_GAP coverage (48-60 Hz) - all available
        51.09991483886941, 48.046775273025006,
        # ALL available HUMAN_PEAK and HUMAN_TAIL coverage (60-85 Hz)
         70.0865804441374, 63.13584519347914,
        # Minimal HIGH_FREQ coverage (85-100 Hz)
        91.4902666875566
    ]
}

# Auto-detect data type and select appropriate resonator grid
def get_resonator_grid(signal_file):
    """
    Automatically select the appropriate resonator grid based on the signal file name
    """
    file_name = str(signal_file).lower()
    if 'human' in file_name:
        print("Detected HUMAN data - using human-optimized resonator grid")
        return clk_resonators_human
    elif 'car' in file_name:
        print("Detected CAR data - using car-optimized resonator grid")
        return clk_resonators_car
    else:
        print("Unknown data type - defaulting to car resonator grid")
        return clk_resonators_car

# Frequency bands for analysis
bands = {
    'LOW_FREQ': (20, 30),
    'CAR_APPROACH': (30, 34),
    'CAR_PEAK': (34, 40),
    'CAR_TAIL': (40, 48),
    'MID_GAP': (48, 60),
    'HUMAN_PEAK': (60, 70),
    'HUMAN_TAIL': (70, 85),
    'HIGH_FREQ': (85, 100)
}

import time

def save_plot(name=None):
    """Save the current figure to a file with a unique name"""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    if name is None:
        name = f"plot_{int(time.time())}.png"
    else:
        name = f"{name}.png"
    
    # Save the figure
    filepath = os.path.join(output_dir, name)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")
    return filepath

def normalize_signal(signal):
    """Normalize signal to [-1, 1] range"""
    signal_min, signal_max = np.min(signal), np.max(signal)
    if signal_max > signal_min:
        return 2 * (signal - signal_min) / (signal_max - signal_min) - 1
    return np.zeros_like(signal)

def load_and_prepare_data(file_path, sampling_freq=1000, duration=None):
    """
    Load data from CSV file and prepare it for analysis
    """
    try:
        # Load data
        data = pd.read_csv(file_path)

        # Use appropriate column for signal data
        if 'amplitude' in data.columns:
            signal = data['amplitude'].values
        else:
            # Use the second column assuming first is time
            signal = data.iloc[:, 1].values

        # Normalize the signal
        signal = normalize_signal(signal)

        # Create time axis
        time = np.arange(len(signal)) / sampling_freq

        # Trim to specified duration if provided
        if duration is not None and duration < time[-1]:
            samples = int(duration * sampling_freq)
            signal = signal[:samples]
            time = time[:samples]

        return signal, time

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def resample_signal(f_new, f_source, data):
    """
    Resample signal to match a new frequency
    """
    n_samples_orig = data.shape[0]
    n_samples_new = int(n_samples_orig * f_new / f_source)

    # Resample the signal
    return resample(data, n_samples_new)

def compute_fft_spectrogram(signal, fs, fmin=1, fmax=80, nperseg=1024, noverlap=512):
    """
    Compute and plot FFT spectrogram
    """
    # Compute spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Plot spectrogram
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Signal Spectrogram')
    plt.ylim(fmin, fmax)
    save_plot() 

    return f, t, Sxx

# Function to process a single resonator in parallel
def process_single_resonator(f0, clk_freq, resampled_signal, progress_dict=None, resonator_id=None):
    """
    Process a single resonator with progress tracking
    """
    try:
        # Get closest resonator function
        resonator_func, actual_freq = get_closest_resonator(f0)

        # Create resonator
        my_resonator = resonator_func()
        my_resonator.log_out_spikes(-1)

        # Progress callback function
        def progress_callback(current, total):
            if progress_dict is not None and resonator_id is not None:
                progress_dict[resonator_id] = (current, total)

        # Process signal with progress tracking
        my_resonator.input_full_data(resampled_signal, progress_callback=progress_callback)

        # Get output spikes
        output_spikes = my_resonator.neurons[-1].out_spikes()

        return output_spikes

    except Exception as e:
        return np.array([])

def process_with_resonator_grid_parallel(signal, fs, clk_resonators, duration, num_processes=None):
    """
    Process signal with resonator grid using parallel processing with real-time progress tracking
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Using {num_processes} processes for parallel computation")

    # Prepare all resonator tasks for parallel processing
    tasks = []
    resonator_id = 0
    resonator_weights = {}  # Store actual sample count for each resonator
    
    for clk_freq, freqs in clk_resonators.items():
        print(f"Preparing resonators for clock frequency {clk_freq}")
        # Resample signal to match clock frequency (without windowing)
        sliced_data_resampled = resample_signal(clk_freq, fs, signal)
        actual_samples = len(sliced_data_resampled)
        
        # Add tasks for all resonators at this clock frequency
        for f0 in freqs:
            tasks.append((f0, clk_freq, sliced_data_resampled, resonator_id))
            resonator_weights[resonator_id] = actual_samples  # Store actual work for this resonator
            resonator_id += 1

    print(f"Processing {len(tasks)} resonators in parallel across all clock frequencies")
    
    # Create shared progress dictionary using Manager
    with multiprocessing.Manager() as manager:
        progress_dict = manager.dict()
        stop_event = threading.Event()
        
        # Start progress monitor in a separate thread with weighted progress
        monitor_thread = threading.Thread(
            target=progress_monitor, 
            args=(progress_dict, len(tasks), resonator_weights, stop_event)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Process ALL resonators from ALL clock frequencies in parallel
            results = Parallel(n_jobs=num_processes, verbose=0)(
                delayed(process_single_resonator)(f0, clk_freq, resampled_signal, progress_dict, res_id) 
                for f0, clk_freq, resampled_signal, res_id in tasks
            )
        finally:
            # Stop the progress monitor
            stop_event.set()
            monitor_thread.join(timeout=1)

    # Reorganize results back into the original structure
    output = {}
    result_idx = 0
    
    for clk_freq, freqs in clk_resonators.items():
        output[clk_freq] = []
        for f0 in freqs:
            output[clk_freq].append(results[result_idx])
            result_idx += 1

    return output

def spikes_event_spectrogram(clk_freq, events, window_ms=10, duration_s=None):
    """
    Convert spike events to binned counts
    """
    window = clk_freq / 1000 * window_ms

    if duration_s is None:
        if len(events) == 0:
            return np.array([0])
        duration_s = events[-1] / clk_freq + 1

    duration_samples = int(duration_s * clk_freq)
    N = int(np.ceil(duration_samples / window))

    bins = np.zeros(N, dtype=int)

    if len(events) > 0:
        # Find which bin each event belongs to
        bin_indices = (events // window).astype(int)

        # Only use valid indices
        valid_indices = bin_indices[bin_indices < N]

        # Count events in each bin
        for idx in valid_indices:
            bins[idx] += 1

    return bins

def create_larger_bins(spectrogram, bin_factor=10):
    """
    Create larger time bins by reshaping and taking max within each bin
    Similar to the successful approach in situational awareness detection
    """
    if spectrogram.shape[1] % bin_factor != 0:
        # Pad to make divisible
        pad_size = bin_factor - (spectrogram.shape[1] % bin_factor)
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_size)), mode='edge')
    
    # Reshape and take max within each bin
    reshaped = spectrogram.reshape(spectrogram.shape[0], -1, bin_factor)
    binned_spectrogram = np.max(reshaped, axis=2)
    
    return binned_spectrogram

def events_to_max_spectrogram(resonators_by_clk, duration, clk_resonators, signal_file, main_clk=153600):
    """
    Convert spike events to max spectrogram with DC removal, thresholding, and amplitude enhancement for robust detection
    """
    # Detect data type for adaptive parameters
    is_human_data = 'human' in str(signal_file).lower()
    
    # Get all frequencies from resonator grid
    all_freqs = []
    for clk_freq, freqs in clk_resonators.items():
        all_freqs.extend(freqs)

    # Create empty spectrogram - use 10ms window for fine detail
    max_spikes_spectrogram = np.zeros((len(all_freqs), int(duration * 100)))
    i = 0

    for clk_freq, spikes_arrays in resonators_by_clk.items():
        for events in spikes_arrays:
            # Convert events to binned spike counts
            spikes_spectrogram = spikes_event_spectrogram(clk_freq, events, 10, duration)

            # Ensure we have enough bins
            if len(spikes_spectrogram) > 0:
                # Match dimensions
                if len(spikes_spectrogram) >= max_spikes_spectrogram.shape[1]:
                    max_spikes_spectrogram[i, :] = spikes_spectrogram[:max_spikes_spectrogram.shape[1]]
                else:
                    max_spikes_spectrogram[i, :len(spikes_spectrogram)] = spikes_spectrogram

                # Apply clock frequency normalization
                max_spikes_spectrogram[i] *= main_clk / clk_freq
                
                # Remove DC component for better contrast
                max_spikes_spectrogram[i] -= np.mean(max_spikes_spectrogram[i])
                
                # THRESHOLD: Set negative values to zero
                max_spikes_spectrogram[i][max_spikes_spectrogram[i] < 0] = 0
                
                # AMPLITUDE ENHANCEMENT: Apply data-specific power function
                if np.max(max_spikes_spectrogram[i]) > 0:
                    # Normalize to [0,1] then apply power function for enhancement
                    normalized = max_spikes_spectrogram[i] / np.max(max_spikes_spectrogram[i])
                    if is_human_data:
                        enhanced = np.power(normalized, 0.6)  # Less aggressive for human data
                    else:
                        enhanced = np.power(normalized, 0.5)  # More aggressive for car data
                    max_spikes_spectrogram[i] = enhanced * np.max(max_spikes_spectrogram[i])

            i += 1

    return max_spikes_spectrogram, all_freqs


def spikes_to_bands(spectrogram, frequencies):
    """
    Group spike spectrogram into frequency bands using max for stronger signal visibility.
    """
    # Use original frequencies without correction
    corrected_frequencies = np.array(frequencies)

    # Create band spectrogram
    bands_spectrogram = np.zeros((len(bands), spectrogram.shape[1]))

    # Fill with data for each band
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        # Find indices of frequencies in this band
        band_indices = np.where((corrected_frequencies >= fmin) & (corrected_frequencies < fmax))[0]

        if len(band_indices) > 0:
            # Use max for stronger signal visibility (now safe with thresholding)
            bands_spectrogram[i] = np.max(spectrogram[band_indices], axis=0)

    return bands_spectrogram

def plot_resonator_output(events, clk_freq, duration, frequency, show_spikes=True):
    """
    Plot spikes emitted by a resonator
    """
    if len(events) == 0:
        print(f"No spikes detected for {frequency} Hz resonator")
        if show_spikes:
            plt.figure(figsize=(12, 4))
            plt.title(f'Resonator Output for {frequency} Hz - No spikes detected')
            plt.ylabel('Spike Count')
            plt.xlabel('Time (s)')
            plt.grid(True)
            save_plot()
        return 0, 0

    # Create spike histogram with 100ms bins
    bins = np.linspace(0, duration, int(duration * 10))
    spike_times = events / clk_freq
    spike_counts, bin_edges = np.histogram(spike_times, bins=bins)

    if show_spikes:
        plt.figure(figsize=(12, 4))
        plt.bar(bin_edges[:-1], spike_counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.7)
        plt.title(f'Resonator Output for {frequency} Hz')
        plt.ylabel('Spike Count')
        plt.xlabel('Time (s)')
        plt.grid(True)
        save_plot()

    # Return statistics
    return np.max(spike_counts), np.std(spike_counts)

def visualize_comparison(signal, time, f, t, Sxx, spikes_bands_spectrogram, duration, signal_file, 
                        file_boundaries=None, boundary_labels=None):
    """
    Create comprehensive visualization comparing raw signal, FFT and resonator responses
    """
    # Detect data type for adaptive visualization parameters
    is_human_data = 'human' in str(signal_file).lower()
    
    # Create a figure with 3 rows
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1.5, 1.5]})

    # Plot 1: Raw Signal
    axs[0].plot(time, signal)
    axs[0].set_title('Raw Signal', fontsize=14)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, alpha=0.3)
    
    # Add file boundaries to raw signal plot
    if file_boundaries is not None and len(file_boundaries) > 0:
        for idx, boundary_time in enumerate(file_boundaries):
            label = boundary_labels[idx] if boundary_labels and idx < len(boundary_labels) else 'File boundary'
            axs[0].axvline(x=boundary_time, color='red', linestyle='--', 
                          alpha=0.7, label=label if idx == 0 else '')
        if boundary_labels:
            axs[0].legend()

    # Plot 2: FFT Spectrogram
    # Create band labels
    band_labels = [f'{fmin}-{fmax} ({band})' for band, (fmin, fmax) in bands.items()]

    # Convert FFT spectrogram to band-based representation
    fft_bin_spectogram = np.zeros((len(bands), len(t)))
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        # Find frequency indices for this band
        f_indices = np.where((f >= fmin) & (f < fmax))[0]
        if len(f_indices) > 0:
            # Average power in this band
            fft_bin_spectogram[i] = np.mean(Sxx[f_indices], axis=0)

    # Apply log transformation to enhance contrast
    fft_bin_spectogram = 10 * np.log10(fft_bin_spectogram + 1e-10)

    im = axs[1].imshow(fft_bin_spectogram, aspect='auto', cmap='jet', origin='lower',
               extent=[0, duration, 0, len(bands)])
    axs[1].set_yticks(np.arange(len(band_labels)) + 0.5)
    axs[1].set_yticklabels(band_labels)
    axs[1].set_title('(a) FFT Spectrogram', fontsize=14)
    axs[1].set_ylabel('Frequency Band')
    fig.colorbar(im, ax=axs[1], label='Power (dB)', pad=0.01)
    
    # Add file boundaries to FFT spectrogram plot
    if file_boundaries is not None and len(file_boundaries) > 0:
        for boundary_time in file_boundaries:
            axs[1].axvline(x=boundary_time, color='red', linestyle='--', alpha=0.7)

    # Plot 3: Spikegram (Resonator Output)
    # Downsample to match FFT spectrogram time resolution
    target_time_bins = len(t)  # Match FFT spectrogram time resolution
    if spikes_bands_spectrogram.shape[1] > target_time_bins:
        # Reshape to match FFT spectrogram time bins
        factor = spikes_bands_spectrogram.shape[1] // target_time_bins
        if factor > 1:
            reshaped = np.zeros((spikes_bands_spectrogram.shape[0], target_time_bins))
            for i in range(target_time_bins):
                start_idx = i * factor
                end_idx = min((i + 1) * factor, spikes_bands_spectrogram.shape[1])
                if end_idx > start_idx:
                    reshaped[:, i] = np.max(spikes_bands_spectrogram[:, start_idx:end_idx], axis=1)
            spikes_bands_spectrogram = reshaped

    # Set data-specific color limits and enhancement
    if is_human_data:
        vmax = np.percentile(spikes_bands_spectrogram, 92)  # more aggressive for human (95% vs 98%)
        power_factor = 1  # Standard visualization enhancement for human
    else:
        vmax = np.percentile(spikes_bands_spectrogram, 98)  # Standard for car
        power_factor = 0.5  # Standard visualization enhancement for car
    
    vmin = 0
    
    # Apply power transformation to create smoother gradient around peaks
    spikes_for_display = np.copy(spikes_bands_spectrogram)
    # Normalize to [0,1] range first
    if vmax > 0:
        normalized = np.clip(spikes_for_display / vmax, 0, 1)
        # Apply data-specific power function for visualization enhancement
        enhanced = np.power(normalized, power_factor)
        spikes_for_display = enhanced * vmax

    im2 = axs[2].imshow(spikes_for_display, aspect='auto', cmap='jet', origin='lower',
                extent=[0, duration, 0, len(bands)], vmin=vmin, vmax=vmax)
    axs[2].set_yticks(np.arange(len(band_labels)) + 0.5)
    axs[2].set_yticklabels(band_labels)
    axs[2].set_title('(b) Resonator-based Spikegram', fontsize=14)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Frequency Band')
    fig.colorbar(im2, ax=axs[2], label='Spike Activity', pad=0.01)
    
    # Add file boundaries to spikegram plot
    if file_boundaries is not None and len(file_boundaries) > 0:
        for boundary_time in file_boundaries:
            axs[2].axvline(x=boundary_time, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_plot()

def process_and_visualize_single_resonator(output, clk_freq, target_freq, duration, clk_resonators):
    """
    Process and visualize the output of a single resonator
    """
    if clk_freq in output and len(output[clk_freq]) > 0:
        # Find the closest resonator to target frequency
        freqs = clk_resonators[clk_freq]
        closest_idx = np.argmin([abs(f - target_freq) for f in freqs])
        selected_freq = freqs[closest_idx]
        selected_resonator = output[clk_freq][closest_idx]

        # Visualize this resonator
        if len(selected_resonator) > 0:
            stats = plot_resonator_output(
                selected_resonator,
                clk_freq,
                duration,
                selected_freq
            )
            print(f"Resonator {selected_freq} Hz stats: max_count={stats[0]}, std={stats[1]:.2f}")
            return selected_freq, stats

    return None, (0, 0)

def create_combined_analysis(signal_file, fs=1000, duration=10, num_processes=4):
    """
    Create comprehensive analysis comparing FFT and resonator approaches
    """
    print("Starting analysis...")

    # Auto-detect and select appropriate resonator grid
    clk_resonators = get_resonator_grid(signal_file)
    print(f"Using {len(clk_resonators[153600])} resonators for analysis")

    # Load and prepare data
    signal, time = load_and_prepare_data(signal_file, fs, duration)
    if signal is None:
        print("Failed to load signal data")
        return None

    print(f"Loaded signal with {len(signal)} samples, {time[-1]:.2f} seconds")

    # Update duration based on loaded data
    duration = time[-1]

    # Plot raw signal
    plt.figure(figsize=(14, 4))
    plt.plot(time, signal)
    plt.title('Raw Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    save_plot()

    # Compute FFT spectrogram
    print("Computing FFT spectrogram...")
    f, t, Sxx = compute_fft_spectrogram(signal, fs, fmin=1, fmax=64)

    # Process with resonator grid using parallel implementation
    print("Processing with resonator grid using parallel implementation...")
    output = process_with_resonator_grid_parallel(
        signal,
        fs,
        clk_resonators,
        duration,
        num_processes=num_processes
    )

    # Create spike spectrograms
    print("Creating spike spectrograms...")
    max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
        output,
        duration,
        clk_resonators,
        signal_file
    )

    # Group by frequency bands
    print("Grouping by frequency bands...")
    spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)

    # Visualize individual resonator output
    print("Visualizing individual resonator outputs...")

    # Show different resonator based on data type
    if 'human' in str(signal_file).lower():
        # For human data, show 69.4 Hz resonator (human peak)
        process_and_visualize_single_resonator(output, 153600, 69.4, duration, clk_resonators)
    else:
        # For car data, show 40.2 Hz resonator (car peak)
        process_and_visualize_single_resonator(output, 153600, 40.2, duration, clk_resonators)

    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    visualize_comparison(
        signal,
        time,
        f, t, Sxx,
        spikes_bands_spectrogram,
        duration,
        signal_file,
        file_boundaries=None,  # Single file analysis has no boundaries
        boundary_labels=None
    )

    print("Analysis completed")

    return {
        'signal': signal,
        'time': time,
        'fft_spectogram': (f, t, Sxx),
        'resonator_outputs': output,
        'max_spikes_spectrogram': max_spikes_spectrogram,
        'spikes_bands_spectrogram': spikes_bands_spectrogram,
        'duration': duration
    }

def combine_resonator_outputs(outputs_list, durations_list):
    """Combine resonator outputs from multiple files"""
    combined_output = {}
    
    # Initialize output structure using the selected resonator grid
    sample_output = outputs_list[0]
    for clk_freq in sample_output.keys():
        combined_output[clk_freq] = []
        num_resonators = len(sample_output[clk_freq])
        for i in range(num_resonators):
            combined_output[clk_freq].append(np.array([]))

    cumulative_duration = 0

    for file_idx, (output, duration) in enumerate(zip(outputs_list, durations_list)):
        print(f"Combining output from file {file_idx + 1}, duration: {duration:.2f}s")

        for clk_freq in output.keys():
            for resonator_idx, spikes in enumerate(output[clk_freq]):
                if resonator_idx < len(combined_output[clk_freq]) and len(spikes) > 0:
                    # Adjust spike times for concatenation
                    adjusted_spikes = spikes + int(cumulative_duration * clk_freq)

                    if len(combined_output[clk_freq][resonator_idx]) > 0:
                        combined_output[clk_freq][resonator_idx] = np.concatenate([
                            combined_output[clk_freq][resonator_idx],
                            adjusted_spikes
                        ])
                    else:
                        combined_output[clk_freq][resonator_idx] = adjusted_spikes

        cumulative_duration += duration

    print(f"Combined total duration: {cumulative_duration:.2f}s")
    return combined_output, cumulative_duration

def analyze_multiple_files(file_paths, label, duration_per_file=60, num_processes=15):
    """Analyze multiple files and combine them"""
    print(f"\n==== ANALYZING {label.upper()} DATA ====")

    individual_outputs = []
    individual_signals = []
    individual_times = []
    individual_durations = []

    for i, file_path in enumerate(file_paths):
        print(f"\n--- Processing file {i+1}/{len(file_paths)}: {file_path} ---")

        # Auto-detect and select appropriate resonator grid
        clk_resonators = get_resonator_grid(file_path)
        
        # Load and prepare data
        signal, time = load_and_prepare_data(file_path, 1000, duration_per_file)
        
        if signal is None or len(signal) == 0:
            print(f"Failed to load signal data from {file_path}")
            continue

        duration = time[-1]
        print(f"Loaded signal with {len(signal)} samples, {duration:.2f} seconds")

        individual_signals.append(signal)
        individual_times.append(time)
        individual_durations.append(duration)

        # Plot individual signal
        plt.figure(figsize=(14, 3))
        plt.plot(time, signal)
        plt.title(f'{label} Signal - File {i+1}: {file_path.name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True, alpha=0.3)
        save_plot(f"{label}_signal_file_{i+1}")

        print(f"Processing file {i+1} with resonator grid...")
        try:
            output = process_with_resonator_grid_parallel(
                signal,
                1000,
                clk_resonators,
                duration,
                num_processes=num_processes
            )
            individual_outputs.append(output)
            print(f"Successfully processed file {i+1}")
        except Exception as e:
            print(f"ERROR processing file {i+1}: {e}")
            individual_outputs.append({})

    if not individual_outputs:
        print("Failed to process any files")
        return None

    print("\nCombining resonator outputs...")
    combined_resonator_output, total_duration = combine_resonator_outputs(
        individual_outputs, individual_durations
    )

    # Combine signals for visualization
    combined_signal = np.concatenate(individual_signals)
    combined_time = np.arange(len(combined_signal)) / 1000

    plt.figure(figsize=(14, 4))
    plt.plot(combined_time, combined_signal)
    plt.title(f'Combined {label} Signal (Total: {total_duration:.1f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)

    # Mark boundaries between files
    cumulative_time = 0
    for idx, dur in enumerate(individual_durations[:-1]):
        cumulative_time += dur
        plt.axvline(x=cumulative_time, color='red', linestyle='--',
                   alpha=0.7, label='File boundary' if idx == 0 else '')
    
    if len(individual_durations) > 1:
        plt.legend()
    save_plot(f"{label}_combined_signal")

    print("Computing FFT spectrogram...")
    f, t, Sxx = compute_fft_spectrogram(
        combined_signal, 1000, fmin=1, fmax=100
    )

    # Create spike spectrograms using the same resonator grid as the last file
    print("Creating spike spectrograms...")
    max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
        combined_resonator_output,
        total_duration,
        clk_resonators,
        file_paths[0]  # Use first file for signal type detection
    )

    # Group by frequency bands
    print("Grouping by frequency bands...")
    spikes_bands_spectrogram = spikes_to_bands(max_spikes_spectrogram, all_freqs)

    # Visualize individual resonator output (use appropriate frequency for signal type)
    print("Visualizing individual resonator outputs...")
    if 'human' in str(file_paths[0]).lower():
        # For human data, show 69.4 Hz resonator (human peak)
        process_and_visualize_single_resonator(combined_resonator_output, 153600, 69.4, total_duration, clk_resonators)
    else:
        # For car data, show 40.2 Hz resonator (car peak)
        process_and_visualize_single_resonator(combined_resonator_output, 153600, 40.2, total_duration, clk_resonators)

    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    
    # Calculate file boundaries for visualization
    file_boundaries = []
    cumulative_time = 0
    for dur in individual_durations[:-1]:  # Don't include the last boundary
        cumulative_time += dur
        file_boundaries.append(cumulative_time)
    
    # Create boundary labels based on file type
    if 'human' in str(file_paths[0]).lower():
        boundary_labels = ['human/human_nothing']
    else:
        boundary_labels = ['car/car_nothing']
    
    visualize_comparison(
        combined_signal,
        combined_time,
        f, t, Sxx,
        spikes_bands_spectrogram,
        total_duration,
        file_paths[0],  # Use first file for signal type detection
        file_boundaries=file_boundaries,
        boundary_labels=boundary_labels
    )

    print(f"{label} analysis completed")
    return {
        'signal': combined_signal,
        'time': combined_time,
        'resonator_outputs': combined_resonator_output,
        'duration': total_duration,
        'max_spikes_spectrogram': max_spikes_spectrogram,
        'spikes_bands_spectrogram': spikes_bands_spectrogram
    }

def analyze_all_files(duration_per_file=60, num_processes=15):
    """Analyze all 4 files: car.csv, car_nothing.csv, human.csv, human_nothing.csv"""
    
    # Process human data (footsteps)
    human_file_paths = [
        DATA_DIR / "human.csv",
        DATA_DIR / "human_nothing.csv"
    ]
    
    human_results = analyze_multiple_files(
        human_file_paths,
        label="Human",
        duration_per_file=duration_per_file,
        num_processes=num_processes
    )

    # Process car data (vehicle vibrations)
    car_file_paths = [
        DATA_DIR / "car.csv",
        DATA_DIR / "car_nothing.csv"
    ]
    
    car_results = analyze_multiple_files(
        car_file_paths,
        label="Car",
        duration_per_file=duration_per_file,
        num_processes=num_processes
    )

    print("\n=== ALL FILES ANALYSIS COMPLETED ===")
    
    if human_results and car_results:
        print(f"Human analysis: {human_results['duration']:.1f}s total")
        print(f"Car analysis: {car_results['duration']:.1f}s total")
    
    return human_results, car_results

def progress_monitor(progress_dict, total_resonators, resonator_weights, stop_event):
    """
    Monitor progress across all parallel resonator processes with weighted progress based on actual work
    """
    start_time = time.time()
    
    # Calculate total work (sum of all samples across all resonators)
    total_work = sum(resonator_weights.values())
    
    print(f"\nProcessing {total_resonators} resonators in parallel:")
    for resonator_id, samples in resonator_weights.items():
        clk_freq = 1536000 if samples > 50000 else 153600  # Estimate clock freq from sample count
        print(f"  Resonator {resonator_id}: {samples:,} samples @ {clk_freq} Hz")
    print(f"Total work: {total_work:,} samples")
    
    last_percent = -1
    
    while not stop_event.is_set():
        time.sleep(0.5)  # Check every 0.5 seconds
        
        # Calculate weighted progress
        completed_work = 0
        completed_resonators = 0
        
        for resonator_id in range(total_resonators):
            if resonator_id in progress_dict:
                current, total_for_resonator = progress_dict[resonator_id]
                resonator_weight = resonator_weights[resonator_id]
                
                # Calculate work completed by this resonator
                if total_for_resonator > 0:
                    resonator_progress = min(current / total_for_resonator, 1.0)
                    completed_work += resonator_progress * resonator_weight
                    
                    if resonator_progress >= 1.0:
                        completed_resonators += 1
        
        # Calculate overall percentage based on weighted work
        if total_work > 0:
            percent = int((completed_work / total_work) * 100)
        else:
            percent = 0
        
        # Only print when percentage changes
        if percent != last_percent:
            elapsed = time.time() - start_time
            
            if percent > 0:
                eta_seconds = (elapsed / percent) * (100 - percent)
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "calculating..."
            
            # Create progress bar
            bar_length = 40
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\r[{bar}] {percent:3d}% | {completed_work:,.0f}/{total_work:,} samples | "
                  f"Elapsed: {int(elapsed//60)}:{int(elapsed%60):02d} | ETA: {eta_str}", end='', flush=True)
            
            last_percent = percent
        
        # Check if we're done (all work completed)
        if percent >= 100:
            break
    
    # Final progress update
    elapsed = time.time() - start_time
    print(f"\r[{'█' * 40}] 100% | {total_work:,}/{total_work:,} samples | "
          f"Elapsed: {int(elapsed//60)}:{int(elapsed%60):02d} | Complete!         ")
    print()

def create_classification_features(spikes_bands_spectrogram, duration, signal_file, 
                                segment_length=300, overlap_ratio=0.5):
    """
    Create classification-ready features with adaptive binning based on data type
    """
    is_human_data = 'human' in str(signal_file).lower()
    
    # Define adaptive binning strategy
    if is_human_data:
        bin_sizes = [5, 15, 30]  # seconds - for event-based detection
        print("Using human-optimized binning: 5s, 15s, 30s windows")
    else:
        bin_sizes = [10, 30, 60]  # seconds - for periodic detection  
        print("Using car-optimized binning: 10s, 30s, 60s windows")
    
    # Enhanced frequency bands for classification
    classification_bands = {
        'LOW_NOISE': (1, 20),
        'CAR_ENGINE': (25, 35), 
        'CAR_VIBRATION': (35, 50),
        'MID_TRANSITION': (50, 60),
        'HUMAN_FOOTSTEP': (60, 80),
        'HIGH_IMPACT': (80, 100),
    }
    
    features = {}
    
    # Create segments
    step_size = int(segment_length * (1 - overlap_ratio))
    n_segments = max(1, int((duration - segment_length) / step_size) + 1)
    
    print(f"Creating {n_segments} segments of {segment_length}s each")
    
    for segment_idx in range(n_segments):
        start_time = segment_idx * step_size
        end_time = min(start_time + segment_length, duration)
        
        # Convert time to sample indices (assuming 10ms bins)
        start_sample = int(start_time * 100)  # 100 samples per second
        end_sample = int(end_time * 100)
        
        segment_data = spikes_bands_spectrogram[:, start_sample:end_sample]
        
        segment_features = {}
        
        # Multi-scale temporal binning
        for bin_size in bin_sizes:
            bin_samples = int(bin_size * 100)  # Convert to samples
            n_bins = segment_data.shape[1] // bin_samples
            
            if n_bins > 0:
                # Reshape and aggregate
                trimmed_data = segment_data[:, :n_bins * bin_samples]
                reshaped = trimmed_data.reshape(trimmed_data.shape[0], n_bins, bin_samples)
                
                # Statistical features for each bin size
                segment_features[f'mean_{bin_size}s'] = np.mean(reshaped, axis=2)
                segment_features[f'max_{bin_size}s'] = np.max(reshaped, axis=2)
                segment_features[f'std_{bin_size}s'] = np.std(reshaped, axis=2)
                
                # Activity detection features
                activity_threshold = np.percentile(reshaped, 90)
                segment_features[f'activity_ratio_{bin_size}s'] = np.mean(reshaped > activity_threshold, axis=2)
        
        # Frequency band energy ratios
        total_energy = np.sum(segment_data, axis=0)
        for band_name in ['CAR_ENGINE', 'CAR_VIBRATION', 'HUMAN_FOOTSTEP']:
            band_indices = []  # Map band names to indices in your frequency bands
            if len(band_indices) > 0:
                band_energy = np.sum(segment_data[band_indices], axis=0)
                segment_features[f'{band_name}_ratio'] = np.mean(band_energy / (total_energy + 1e-10))
        
        # Periodicity features (for car detection)
        if not is_human_data:
            # FFT of activity pattern to detect periodicity
            activity_pattern = np.mean(segment_data, axis=0)
            fft_activity = np.abs(np.fft.fft(activity_pattern))
            
            # Look for peaks around expected car frequencies (every 50s = 0.02 Hz)
            freqs = np.fft.fftfreq(len(activity_pattern), d=0.01)  # 10ms sampling
            target_freq = 0.02  # 50-second period
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            segment_features['periodicity_strength'] = fft_activity[freq_idx]
        
        features[f'segment_{segment_idx}'] = segment_features
    
    return features

def create_dataset_for_classification(human_results, car_results):
    """
    Create a structured dataset ready for machine learning classification
    """
    dataset = []
    labels = []
    
    # Process human data
    if human_results:
        human_features = create_classification_features(
            human_results['spikes_bands_spectrogram'],
            human_results['duration'],
            'human.csv'
        )
        
        for segment_name, features in human_features.items():
            # Flatten all features into a single vector
            feature_vector = []
            for feature_name, feature_data in features.items():
                if hasattr(feature_data, 'flatten'):
                    feature_vector.extend(feature_data.flatten())
                else:
                    feature_vector.append(feature_data)
            
            dataset.append(feature_vector)
            labels.append('human')
    
    # Process car data
    if car_results:
        car_features = create_classification_features(
            car_results['spikes_bands_spectrogram'],
            car_results['duration'],
            'car.csv'
        )
        
        for segment_name, features in car_features.items():
            # Flatten all features into a single vector
            feature_vector = []
            for feature_name, feature_data in features.items():
                if hasattr(feature_data, 'flatten'):
                    feature_vector.extend(feature_data.flatten())
                else:
                    feature_vector.append(feature_data)
            
            dataset.append(feature_vector)
            labels.append('car')
    
    return np.array(dataset), np.array(labels)

def run_complete_analysis_with_snn_classification(duration_per_file=130, num_processes=15):
    """
    Complete analysis pipeline with SNN classification
    """
    print("🚀 STARTING COMPLETE RESONATOR + SNN CLASSIFICATION PIPELINE")
    print("=" * 70)
    
    # Step 1: Analyze all 4 files with resonators
    print("\n📊 Step 1: Processing signals with resonator grids...")
    human_results, car_results = analyze_all_files(
        duration_per_file=duration_per_file,
        num_processes=num_processes
    )
    
    if not human_results or not car_results:
        print("❌ Failed to process signals with resonators")
        return None
    
    print(f"✅ Resonator processing completed")
    print(f"   Human data: {human_results['duration']:.1f}s processed")
    print(f"   Car data: {car_results['duration']:.1f}s processed")
    
    """# Step 2: Run SNN Classification
    print("\n🧠 Step 2: Training SNN for classification...")
    
    try:
        # Import the SNN classification functions
        from snn_classification import run_enhanced_car_classification, GeophoneSNN, analyze_spectrograms_for_segments
        
        # Create SNN classifier
        snn = GeophoneSNN(n_hidden=40, learning_rate=0.015)
        
        # Process car data for classification
        print("🚗 Processing car data for SNN classification...")
        
        # Analyze car spikegram for segments
        car_segments, car_seg_labels, car_confidence = analyze_spectrograms_for_segments(
            car_results['spikes_bands_spectrogram'], 
            car_results['duration'], 
            'car'
        )
        
        # Get car segments (only those with car activity)
        car_indices = car_seg_labels == 1
        car_data_segments = car_segments[car_indices]
        car_data_labels = [0] * np.sum(car_indices)  # 0 = car
        
        print(f"   Found {np.sum(car_indices)} car segments out of {len(car_seg_labels)} total")
        
        # For car_nothing, we need to process that file separately
        print("🔇 Processing car_nothing data...")
        car_nothing_file = DATA_DIR / "car_nothing.csv"
        car_nothing_signal, car_nothing_time = load_and_prepare_data(car_nothing_file, 1000, duration_per_file)
        
        if car_nothing_signal is not None:
            # Process with resonators
            clk_resonators = get_resonator_grid(car_nothing_file)
            car_nothing_output = process_with_resonator_grid_parallel(
                car_nothing_signal, 1000, clk_resonators, car_nothing_time[-1], num_processes=10
            )
            
            # Create spikegrams
            car_nothing_spikes_spec, car_nothing_freqs = events_to_max_spectrogram(
                car_nothing_output, car_nothing_time[-1], clk_resonators, car_nothing_file
            )
            car_nothing_bands_spec = spikes_to_bands(car_nothing_spikes_spec, car_nothing_freqs)
            
            # Analyze for nothing segments
            nothing_segments, nothing_seg_labels, nothing_confidence = analyze_spectrograms_for_segments(
                car_nothing_bands_spec, car_nothing_time[-1], 'car'
            )
            
            # Get nothing segments (those with no significant activity)
            nothing_indices = nothing_seg_labels == 0
            nothing_data_segments = nothing_segments[nothing_indices]
            nothing_data_labels = [1] * np.sum(nothing_indices)  # 1 = nothing
            
            print(f"   Found {np.sum(nothing_indices)} nothing segments out of {len(nothing_seg_labels)} total")
        else:
            print("❌ Failed to load car_nothing data")
            nothing_data_segments = []
            nothing_data_labels = []
        
        # Combine all segments for training
        if len(car_data_segments) > 0 and len(nothing_data_segments) > 0:
            all_segments = np.vstack([car_data_segments, nothing_data_segments])
            all_labels = np.array(car_data_labels + nothing_data_labels)
            
            print(f"\n📈 Dataset prepared:")
            print(f"   Total segments: {len(all_segments)}")
            print(f"   Features per segment: {all_segments.shape[1]}")
            print(f"   Car segments: {np.sum(all_labels == 0)}")
            print(f"   Nothing segments: {np.sum(all_labels == 1)}")
            
            # Split into train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                all_segments, all_labels, test_size=0.25, random_state=42, stratify=all_labels
            )
            
            print(f"   Training samples: {len(X_train)}")
            print(f"   Test samples: {len(X_test)}")
            
            # Train SNN
            print("\n🎯 Training SNN...")
            training_history = snn.train(X_train, y_train, n_epochs=80, spike_duration=200)
            
            # Evaluate
            print("\n📊 Evaluating SNN...")
            accuracy, report, cm = snn.evaluate(X_test, y_test)
            
            # Save model
            model_path = os.path.join(os.path.dirname(__file__), "resonator_snn_model.pkl")
            snn.save_model(model_path)
            
            print(f"\n✅ SNN Classification Complete!")
            print(f"📊 Final Results:")
            print(f"   Test Accuracy: {accuracy:.1%}")
            print(f"   Model saved: {model_path}")
            
            # Create comprehensive results
            classification_results = {
                'snn_model': snn,
                'test_accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'training_history': training_history,
                'car_segments': np.sum(car_indices),
                'nothing_segments': np.sum(nothing_indices) if len(nothing_data_segments) > 0 else 0
            }
            
            return {
                'human_results': human_results,
                'car_results': car_results,
                'classification_results': classification_results
            }
        else:
            print("❌ Insufficient data for classification")
            return {
                'human_results': human_results,
                'car_results': car_results,
                'classification_results': None
            }
            
    except Exception as e:
        print(f"❌ SNN Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'human_results': human_results,
            'car_results': car_results,
            'classification_results': None
        }"""

# Example usage
if __name__ == "__main__":
    # You must have RESONATOR_FUNCTIONS defined globally before using this code

    # Run complete analysis with SNN classification
    results = run_complete_analysis_with_snn_classification(
        duration_per_file=240,  # Process 240 seconds from each file
        num_processes=15
    )
    
    if results and results['classification_results']:
        class_results = results['classification_results']
        print(f"\n🎯 FINAL PIPELINE RESULTS:")
        print(f"   Resonator Processing: ✅ Complete")
        print(f"   SNN Classification: ✅ Complete")
        print(f"   Test Accuracy: {class_results['test_accuracy']:.1%}")
        print(f"   Car Segments Detected: {class_results['car_segments']}")
        print(f"   Nothing Segments Detected: {class_results['nothing_segments']}")
        print(f"   Model Architecture: {class_results['snn_model'].n_input_neurons} → {class_results['snn_model'].n_hidden} → 3 neurons")
    else:
        print("\n❌ Pipeline completed with errors")
    
    # Optional: Single file analysis (commented out)
    #results = create_combined_analysis(
    #    signal_file= DATA_DIR / "human.csv",  # Path to your signal file
    #    duration=10,
    #    num_processes=15
    #)