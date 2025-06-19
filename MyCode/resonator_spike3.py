from pathlib import Path
DATA_DIR = Path.home() / "data"

import pandas as pd
import numpy as np
import matplotlib          
matplotlib.use("QtAgg")    
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt, resample, welch
from scipy.stats import kurtosis
import math
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Import predefined resonator functions from module
from sctn.resonator_functions import RESONATOR_FUNCTIONS, get_closest_resonator

# Define frequency bands for geophone signals
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


# Clock frequencies optimized for geophone frequency ranges
SNN_CLK_FREQ_MID = 153600   # for 10-100 Hz (vehicles)

# Resonator frequencies with enhanced coverage for geophone signals
clk_resonators = {
    15360: [1.05, 1.10, 1.15, 1.28, 1.30, 1.59, 1.60, 1.66, 1.75, 1.90, 1.95, 
            2.21, 2.50, 2.68, 2.79, 2.88, 3.05, 3.39, 3.47, 3.72, 4.02, 4.12, 
            4.36, 4.62, 4.77, 5.09, 5.26, 5.45, 5.87, 6.36, 6.94, 7.63, 8.98, 9.54],
    153600: [10.5, 11.0, 11.5, 12.8, 13.0, 15.9, 16.0, 16.6, 17.5, 19.0, 19.5,
             22.1, 25.0, 26.8, 27.9, 28.8, 30.5, 33.9, 34.7, 37.2, 40.2, 41.2,
             43.6, 46.2, 47.7, 50.9, 52.6, 54.5, 58.7, 63.6, 69.4, 76.3, 89.8, 95.4]
}

# Get all frequencies in sorted order
all_resonator_freqs = sorted(sum(clk_resonators.values(), []))

def compute_signal_characteristics(signal, fs):
    """Compute signal characteristics for adaptive processing"""
    # Kurtosis to distinguish impulsive (footsteps) from continuous (vehicle) signals
    kurt = kurtosis(signal)
    
    # Power spectral density to identify dominant frequencies
    freqs, psd = welch(signal, fs, nperseg=min(fs, len(signal)))
    
    # Find dominant frequency
    dominant_freq = freqs[np.argmax(psd)]
    
    # Compute spectral centroid
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    
    return {
        'kurtosis': kurt,
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'is_impulsive': kurt > 5,  # Threshold for impulsive signals
        'is_continuous': kurt < 3   # Threshold for continuous signals
    }

def adaptive_normalize_signal(signal, signal_chars):
    """Adaptive normalization based on signal characteristics"""
    # Remove DC offset
    signal = signal - np.mean(signal)
    
    if signal_chars['is_impulsive']:
        # For impulsive signals (footsteps), preserve peak structure
        # Use percentile-based normalization to preserve impulses
        p95 = np.percentile(np.abs(signal), 95)
        if p95 > 0:
            signal = signal / (2 * p95)
            signal = np.clip(signal, -1, 1)
    else:
        # For continuous signals (vehicles), use standard normalization
        signal_std = np.std(signal)
        if signal_std > 0:
            signal = signal / (3 * signal_std)  # 3-sigma normalization
            signal = np.clip(signal, -1, 1)
    
    return signal

def load_and_prepare_data(file_path, sampling_freq=1000, duration=None, apply_filter=True):
    """Load data from CSV file with geophone-specific preprocessing"""
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            return None, None, None

        print(f"Reading file: {file_path}")
        data = pd.read_csv(file_path)

        # Use amplitude column
        if 'amplitude' in data.columns:
            signal = data['amplitude'].values
        else:
            signal = data.iloc[:, 1].values

        print(f"Raw signal stats - min: {np.min(signal):.4f}, max: {np.max(signal):.4f}, "
              f"mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}")

        # Compute signal characteristics before normalization
        signal_chars = compute_signal_characteristics(signal, sampling_freq)
        print(f"Signal characteristics: {signal_chars}")

        # Apply high-pass filter to remove low-frequency drift
        if apply_filter:
            # High-pass at 0.5 Hz to remove DC and very low frequency drift
            nyq = sampling_freq / 2
            b, a = butter(4, 0.5 / nyq, btype='highpass')
            signal = filtfilt(b, a, signal)

        # Adaptive normalization based on signal type
        signal = adaptive_normalize_signal(signal, signal_chars)

        # Create time axis
        time = np.arange(len(signal)) / sampling_freq

        # Trim to specified duration if provided
        if duration is not None and duration < time[-1]:
            samples = int(duration * sampling_freq)
            signal = signal[:samples]
            time = time[:samples]

        return signal, time, signal_chars

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None, None

def resample_signal(f_new, f_source, data):
    """Resample signal with anti-aliasing filter"""
    if f_new == f_source:
        return data
    
    n_samples_orig = data.shape[0]
    n_samples_new = int(n_samples_orig * f_new / f_source)
    
    # Apply anti-aliasing filter before downsampling
    if f_new < f_source:
        nyq = f_source / 2
        cutoff = f_new / 2
        b, a = butter(8, cutoff / nyq, btype='low')
        data = filtfilt(b, a, data)
    
    return resample(data, n_samples_new)

def compute_fft_spectrogram(signal, fs, fmin=0.1, fmax=80, nperseg=None, noverlap=0.75, title="Signal Spectrogram"):
    """Compute FFT spectrogram with parameters optimized for geophone signals"""
    if nperseg is None:
        nperseg = int(fs * 1.0)  # 1 second window
    
    # Use 75% overlap for better temporal resolution
    noverlap = int(nperseg * noverlap)
    
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                           window='hann', scaling='density')

    plt.figure(figsize=(14, 6))
    
    # Use log scale for better visualization of low-amplitude components
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='jet')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.ylim(fmin, fmax)
    plt.show()

    return f, t, Sxx

def reset_resonator_state(resonator):
    """Completely reset resonator state to prevent contamination"""
    # Reset all neurons in the resonator
    for neuron in resonator.neurons:
        neuron.membrane_potential = 0.0
        neuron.leakage_timer = 0
        neuron.rand_gauss_var = 0
        if hasattr(neuron, 'spike_history'):
            neuron.spike_history = []
        if hasattr(neuron, 'last_spike_time'):
            neuron.last_spike_time = -np.inf
        
    # Reset the resonator's internal state
    resonator.reset_input()
    
    # Clear any logged data
    resonator.forget_logs()

def compute_adaptive_gain(signal_chars, f0, clk_freq):
    """Compute adaptive gain based on signal characteristics and target frequency"""
    base_gain = 1000.0
    
    # Adjust gain based on signal type
    if signal_chars['is_impulsive']:
        # Higher gain for impulsive signals to capture transients
        type_factor = 1.5
    elif signal_chars['is_continuous']:
        # Lower gain for continuous signals to prevent saturation
        type_factor = 0.7
    else:
        type_factor = 1.0
    
    # Frequency-dependent gain adjustment
    # Lower frequencies need higher gain due to geophone response
    if f0 < 5:
        freq_factor = 1.5
    elif f0 < 10:
        freq_factor = 1.2
    elif f0 < 20:
        freq_factor = 1.0
    else:
        freq_factor = 0.8
    
    # Clock frequency adjustment
    clk_factor = np.sqrt(clk_freq / 153600)  # Normalize to higher clock
    
    return base_gain * type_factor * freq_factor * clk_factor

def process_single_resonator(f0, clk_freq, resampled_signal, signal_chars):
    """Process a single resonator with adaptive scaling and state management"""
    try:
        resonator_func, actual_freq = get_closest_resonator(f0)
        
        # Create fresh resonator instance
        my_resonator = resonator_func()
        
        # Complete state reset to prevent contamination
        reset_resonator_state(my_resonator)
        
        # Enable spike logging
        my_resonator.log_out_spikes(-1)
        
        # Compute adaptive gain
        adaptive_gain = compute_adaptive_gain(signal_chars, f0, clk_freq)
        
        # Scale the signal with adaptive gain
        scaled_signal = resampled_signal * adaptive_gain
        
        # Pre-charge resonator with zeros to stabilize
        stabilization_time = 0.1  # 100ms
        stabilization_samples = int(clk_freq * stabilization_time)
        my_resonator.input_full_data(np.zeros(stabilization_samples))
        
        # Reset logs after stabilization
        output_neuron = my_resonator.neurons[-1]
        output_neuron.forget_logs()
        
        # Process the actual signal
        my_resonator.input_full_data(scaled_signal)
        
        # Get output spikes
        output_spikes = output_neuron.out_spikes()
        
        print(f"Resonator {f0:.1f} Hz: {len(output_spikes)} spikes, "
              f"gain: {adaptive_gain:.1f}, actual freq: {actual_freq:.2f}")
        
        return output_spikes

    except Exception as e:
        print(f"Error processing resonator {f0} Hz: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])

def process_with_resonator_grid(signal, fs, clk_resonators, duration, signal_chars, 
                               use_parallel=True, num_processes=None):
    """Process signal with resonator grid using adaptive parameters"""
    if use_parallel and num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    if use_parallel:
        print(f"Using {num_processes} processes for parallel computation")

    output = {}

    for clk_freq, freqs in clk_resonators.items():
        print(f"\nProcessing resonators for clock frequency {clk_freq}")
        output[clk_freq] = []

        # Resample with anti-aliasing
        sliced_data_resampled = resample_signal(clk_freq, fs, signal)

        try:
            if use_parallel:
                results = Parallel(n_jobs=num_processes, verbose=10)(
                    delayed(process_single_resonator)(
                        f0,
                        clk_freq,
                        sliced_data_resampled,
                        signal_chars
                    ) for f0 in freqs
                )
                output[clk_freq] = results
            else:
                for f0 in tqdm(freqs, desc=f"Processing resonators at {clk_freq} Hz"):
                    result = process_single_resonator(f0, clk_freq, sliced_data_resampled, signal_chars)
                    output[clk_freq].append(result)
        except Exception as e:
            print(f"ERROR in processing for {clk_freq} Hz: {e}")

    return output

def adaptive_spike_binning(events, clk_freq, signal_chars, duration_s):
    """Adaptive temporal binning based on signal characteristics"""
    if signal_chars['is_impulsive']:
        # Fine temporal resolution for impulsive signals (5ms bins)
        window_ms = 5
    elif signal_chars['is_continuous']:
        # Coarser resolution for continuous signals (20ms bins)
        window_ms = 20
    else:
        # Default 10ms bins
        window_ms = 10
    
    window_samples = int(clk_freq * window_ms / 1000)
    n_bins = int(np.ceil(duration_s * 1000 / window_ms))
    
    bins = np.zeros(n_bins, dtype=float)
    
    if len(events) > 0:
        # Use histogram for efficient binning
        bin_edges = np.arange(0, (n_bins + 1) * window_samples, window_samples)
        hist, _ = np.histogram(events, bins=bin_edges)
        bins[:len(hist)] = hist
    
    return bins, window_ms

def events_to_max_spectrogram(resonators_by_clk, duration, signal_chars):
    """Convert spike events to spectrogram with adaptive temporal resolution"""
    # Create frequency-to-index mapping
    freq_to_idx = {freq: idx for idx, freq in enumerate(all_resonator_freqs)}
    
    # Determine temporal resolution based on signal type
    base_bin_ms = 10  # Base resolution
    if signal_chars['is_impulsive']:
        bin_ms = 5  # Higher resolution for footsteps
    elif signal_chars['is_continuous']:
        bin_ms = 20  # Lower resolution for vehicles
    else:
        bin_ms = base_bin_ms
    
    n_time_bins = int(duration * 1000 / bin_ms)
    spike_spectrogram = np.zeros((len(all_resonator_freqs), n_time_bins))
    
    # Process each clock frequency group
    for clk_freq, spikes_arrays in resonators_by_clk.items():
        freqs = clk_resonators[clk_freq]
        
        for freq, events in zip(freqs, spikes_arrays):
            if freq in freq_to_idx and len(events) > 0:
                idx = freq_to_idx[freq]
                
                # Adaptive binning
                spike_bins, actual_bin_ms = adaptive_spike_binning(
                    events, clk_freq, signal_chars, duration
                )
                
                # Resample if necessary to match target resolution
                if actual_bin_ms != bin_ms:
                    resample_factor = actual_bin_ms / bin_ms
                    new_length = int(len(spike_bins) / resample_factor)
                    spike_bins = resample(spike_bins, min(new_length, n_time_bins))
                
                # Fill spectrogram
                min_len = min(len(spike_bins), n_time_bins)
                spike_spectrogram[idx, :min_len] = spike_bins[:min_len]
    
    # Frequency-specific normalization
    for i in range(len(all_resonator_freqs)):
        freq = all_resonator_freqs[i]
        row_data = spike_spectrogram[i]
        
        if np.max(row_data) > 0:
            # Compute adaptive baseline
            if freq < 5:  # Low frequency (footsteps)
                # Use median for sparse signals
                baseline = np.median(row_data[row_data > 0]) if np.any(row_data > 0) else 0
            else:  # Higher frequencies
                # Use percentile for continuous signals
                baseline = np.percentile(row_data, 25)
            
            # Subtract baseline and clip negative values
            spike_spectrogram[i] = np.maximum(row_data - baseline, 0)
            
            # Normalize by frequency-specific factor
            if freq < 10:
                norm_factor = np.percentile(spike_spectrogram[i], 95) if np.any(spike_spectrogram[i] > 0) else 1
            else:
                norm_factor = np.max(spike_spectrogram[i])
            
            if norm_factor > 0:
                spike_spectrogram[i] /= norm_factor
    
    return spike_spectrogram, all_resonator_freqs

def spikes_to_bands(spectrogram, frequencies, signal_chars, normalize_band=True):
    """Group spike spectrogram into frequency bands with signal-aware processing"""
    bands_spectrogram = np.zeros((len(bands), spectrogram.shape[1]))
    frequencies = np.array(frequencies)

    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        # Find frequencies in this band
        band_mask = (frequencies >= fmin) & (frequencies < fmax)
        band_indices = np.where(band_mask)[0]
        
        if len(band_indices) > 0:
            # Weight by frequency importance for signal type
            if signal_chars['is_impulsive'] and band_name == 'Delta':
                # Emphasize low frequencies for footsteps
                weights = np.exp(-0.5 * (frequencies[band_indices] - 2)**2)  # Peak at 2 Hz
            elif signal_chars['is_continuous'] and band_name == 'Alpha':
                # Emphasize 10-15 Hz for vehicles
                weights = np.exp(-0.1 * (frequencies[band_indices] - 12.5)**2)  # Peak at 12.5 Hz
            else:
                weights = np.ones(len(band_indices))
            
            # Weighted sum
            for j, idx in enumerate(band_indices):
                bands_spectrogram[i] += spectrogram[idx] * weights[j]
            
            if normalize_band:
                # Normalize by weighted sum of frequencies
                bands_spectrogram[i] /= np.sum(weights)
    
    # Apply band-specific processing
    for i, (band_name, _) in enumerate(bands.items()):
        if band_name == 'Delta' and signal_chars['is_impulsive']:
            # Enhance contrast for footstep detection
            bands_spectrogram[i] = np.power(bands_spectrogram[i], 0.7)
        elif band_name == 'Alpha' and signal_chars['is_continuous']:
            # Smooth for vehicle detection
            from scipy.ndimage import gaussian_filter1d
            bands_spectrogram[i] = gaussian_filter1d(bands_spectrogram[i], sigma=2)
    
    return bands_spectrogram

def plot_bins(Sxx, duration, labels, annotate=False, rotate_annotate=False, show=True,
              colorbar=True, gridlines=True, axs=None, fig=None, vmin=None, vmax=None,
              fontsize=None, title=None):
    """Plot binned spectrogram data"""
    if axs is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        ax = axs

    # Create extent to show time in seconds
    extent = [0, Sxx.shape[1], 0, len(Sxx)]

    im = ax.imshow(Sxx, aspect='auto',
                  cmap='jet', origin='lower',
                  extent=extent,
                  vmin=vmin, vmax=vmax,
                  interpolation='bilinear')  # Smoother interpolation

    # Set y-axis labels
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=fontsize or 12)

    # Set x-axis to show time in seconds
    time_ticks = np.arange(0, Sxx.shape[1] + 1, max(1, Sxx.shape[1] // 10))
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f'{t:.1f}' for t in time_ticks])

    if gridlines:
        # Horizontal grid lines between bands
        ax.set_yticks(np.arange(len(labels) + 1), minor=True)
        ax.yaxis.grid(which='minor', color='grey', linestyle='-', linewidth=1)
        # Vertical grid lines
        ax.xaxis.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=21)

    if colorbar and fig is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.ax.set_ylabel('Normalized Activity', fontsize=12)

    if show:
        plt.show()

    return ax

def visualize_comparison(signal, time, f, t, Sxx, spikes_bands_spectrogram, duration, signal_chars):
    """Create visualization comparing FFT and spike spectrograms"""
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    plt.rcParams["font.family"] = "sans-serif"

    # Create band labels
    band_labels = []
    for band_name, (fmin, fmax) in bands.items():
        if fmin < 1:
            band_labels.append(f'0.1-{fmax} ({band_name})')
        else:
            band_labels.append(f'{fmin}-{fmax} ({band_name})')

    # Plot 1: FFT Spectrogram
    # Compute band power with proper normalization
    fft_bin_spectogram = np.zeros((len(bands), len(t)))
    
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        f_indices = np.where((f >= fmin) & (f < fmax))[0]
        if len(f_indices) > 0:
            # Compute power in band
            band_power = np.sum(Sxx[f_indices], axis=0)
            
            # Log transform for better visualization
            band_power = np.log10(band_power + 1e-10)
            
            # Normalize to [0, 1]
            if np.max(band_power) > np.min(band_power):
                band_power = (band_power - np.min(band_power)) / (np.max(band_power) - np.min(band_power))
            
            fft_bin_spectogram[i] = band_power

    # Ensure time alignment
    time_bins = min(int(duration), fft_bin_spectogram.shape[1])
    fft_bin_spectogram = fft_bin_spectogram[:, :time_bins]

    plot_bins(fft_bin_spectogram, time_bins, band_labels,
              annotate=False, rotate_annotate=True,
              show=False, colorbar=True, gridlines=True,
              fig=fig, axs=axs[0], title='(a) FFT spectrogram')
    axs[0].set_ylabel('Frequency (Hz)', fontsize=18)
    axs[0].tick_params(axis='y', labelsize=14)

    # Plot 2: Spikegram
    # Determine bin size based on signal type
    if signal_chars['is_impulsive']:
        ms_per_bin = 5
    elif signal_chars['is_continuous']:
        ms_per_bin = 20
    else:
        ms_per_bin = 10
    
    bins_per_second = 1000 // ms_per_bin
    
    # Calculate number of complete seconds
    n_seconds = min(spikes_bands_spectrogram.shape[1] // bins_per_second, time_bins)
    
    if n_seconds > 0:
        # Aggregate to 1-second bins for visualization
        spikes_bands_binned = np.zeros((len(bands), n_seconds))
        for i in range(n_seconds):
            start_idx = i * bins_per_second
            end_idx = (i + 1) * bins_per_second
            spikes_bands_binned[:, i] = np.mean(spikes_bands_spectrogram[:, start_idx:end_idx], axis=1)
        
        # Normalize for visualization
        for i in range(len(bands)):
            if np.max(spikes_bands_binned[i]) > 0:
                spikes_bands_binned[i] /= np.max(spikes_bands_binned[i])
    else:
        spikes_bands_binned = np.zeros((len(bands), time_bins))

    plot_bins(spikes_bands_binned, n_seconds, band_labels,
              annotate=False, rotate_annotate=True,
              show=False, colorbar=True, gridlines=True,
              fig=fig, axs=axs[1], title='(b) Spikegram')
    axs[1].set_xlabel('Time (s)', fontsize=18)
    axs[1].set_ylabel('Frequency (Hz)', fontsize=18)
    axs[1].tick_params(axis='y', labelsize=14)

    # Add signal type annotation
    signal_type = "Impulsive (Footsteps)" if signal_chars['is_impulsive'] else \
                  "Continuous (Vehicle)" if signal_chars['is_continuous'] else "Mixed"
    fig.suptitle(f'Signal Type: {signal_type}', fontsize=16)

    plt.tight_layout()
    plt.show()

def create_visualization(signal, time, f, t, Sxx, resonator_output, duration, signal_chars):
    """Create visualization from resonator output"""
    print("Creating spike spectrograms...")
    max_spikes_spectrogram, all_freqs = events_to_max_spectrogram(
        resonator_output,
        duration,
        signal_chars
    )

    print("Grouping by frequency bands...")
    spikes_bands_spectrogram = spikes_to_bands(
        max_spikes_spectrogram, 
        all_freqs, 
        signal_chars,
        normalize_band=True
    )

    print("Creating visualization...")
    visualize_comparison(
        signal,
        time,
        f, t, Sxx,
        spikes_bands_spectrogram,
        duration,
        signal_chars
    )

def combine_resonator_outputs(outputs_list, durations_list, chars_list):
    """Combine resonator outputs from multiple files with proper state management"""
    combined_output = {}
    
    # Initialize output structure
    for clk_freq in clk_resonators.keys():
        combined_output[clk_freq] = []
        num_resonators = len(clk_resonators[clk_freq])
        for i in range(num_resonators):
            combined_output[clk_freq].append(np.array([]))

    cumulative_duration = 0

    for file_idx, (output, duration, signal_chars) in enumerate(zip(outputs_list, durations_list, chars_list)):
        print(f"Combining output from file {file_idx + 1}, duration: {duration:.2f}s, "
              f"type: {'impulsive' if signal_chars['is_impulsive'] else 'continuous'}")

        for clk_freq in clk_resonators.keys():
            if clk_freq in output:
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
    
    # Compute combined signal characteristics
    combined_chars = {
        'is_impulsive': any(c['is_impulsive'] for c in chars_list),
        'is_continuous': any(c['is_continuous'] for c in chars_list),
        'dominant_freq': np.mean([c['dominant_freq'] for c in chars_list]),
        'kurtosis': np.mean([c['kurtosis'] for c in chars_list])
    }
    
    return combined_output, cumulative_duration, combined_chars

def analyze_geophone_data(file_paths, label, duration_per_file=30, num_processes=None):
    """Analyze geophone data with proper signal characterization"""
    print(f"\n==== ANALYZING {label.upper()} DATA ====")

    individual_outputs = []
    individual_signals = []
    individual_times = []
    individual_durations = []
    individual_chars = []

    for i, file_path in enumerate(file_paths):
        print(f"\n--- Processing file {i+1}/{len(file_paths)}: {file_path} ---")

        signal, time, signal_chars = load_and_prepare_data(
            file_path, 1000, duration_per_file, apply_filter=True
        )

        if signal is None or len(signal) == 0:
            print(f"Failed to load signal data from {file_path}")
            continue

        duration = time[-1]
        print(f"Loaded signal with {len(signal)} samples, {duration:.2f} seconds")

        individual_signals.append(signal)
        individual_times.append(time)
        individual_durations.append(duration)
        individual_chars.append(signal_chars)

        # Plot individual signal
        plt.figure(figsize=(14, 3))
        plt.plot(time, signal)
        plt.title(f'{label} Signal - File {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Processing file {i+1} with resonator grid...")
        try:
            output = process_with_resonator_grid(
                signal,
                1000,
                clk_resonators,
                duration,
                signal_chars,
                use_parallel=True,
                num_processes=num_processes
            )
            individual_outputs.append(output)
            print(f"Successfully processed file {i+1}")
        except Exception as e:
            print(f"ERROR processing file {i+1}: {e}")
            import traceback
            traceback.print_exc()
            individual_outputs.append({})

    if not individual_outputs:
        print("Failed to process any files")
        return None

    print("\nCombining resonator outputs...")
    combined_resonator_output, total_duration, combined_chars = combine_resonator_outputs(
        individual_outputs, individual_durations, individual_chars
    )

    # Combine signals for visualization
    combined_signal = np.concatenate(individual_signals)
    combined_time = np.arange(len(combined_signal)) / 1000

    plt.figure(figsize=(14, 4))
    plt.plot(combined_time, combined_signal)
    plt.title(f'Combined {label} Signal')
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
    plt.show()

    print("Computing FFT spectrogram...")
    f, t, Sxx = compute_fft_spectrogram(
        combined_signal, 1000, fmin=0.1, fmax=80,
        title=f"{label} Signal Spectrogram (Duration: {total_duration:.1f}s)"
    )

    print("Creating visualization...")
    create_visualization(
        combined_signal,
        combined_time,
        f, t, Sxx,
        combined_resonator_output,
        total_duration,
        combined_chars
    )

    print(f"{label} analysis completed")
    return {
        'signal': combined_signal,
        'time': combined_time,
        'resonator_outputs': combined_resonator_output,
        'duration': total_duration,
        'signal_chars': combined_chars
    }

def main():
    """Main function to run the geophone signal analysis"""
    num_processes = None  # Will use CPU count - 1

    # Process human data (footsteps)
    human_file_paths = [
        DATA_DIR / "human.csv",
        DATA_DIR / "human_nothing.csv"
    ]
    
    human_results = analyze_geophone_data(
        human_file_paths,
        label="Human",
        duration_per_file=60,
        num_processes=num_processes
    )

    # Process car data (vehicle vibrations)
    car_file_paths = [
        DATA_DIR / "car.csv",
        DATA_DIR / "car_nothing.csv"
    ]
    
    car_results = analyze_geophone_data(
        car_file_paths,
        label="Car",
        duration_per_file=60,
        num_processes=num_processes
    )

    print("\nAll analysis completed!")
    
    # Summary comparison
    if human_results and car_results:
        print("\n=== SIGNAL CHARACTERISTICS SUMMARY ===")
        print(f"Human signals: Kurtosis={human_results['signal_chars']['kurtosis']:.2f}, "
              f"Dominant freq={human_results['signal_chars']['dominant_freq']:.2f} Hz")
        print(f"Car signals: Kurtosis={car_results['signal_chars']['kurtosis']:.2f}, "
              f"Dominant freq={car_results['signal_chars']['dominant_freq']:.2f} Hz")
    
    return human_results, car_results

if __name__ == "__main__":
    main()
