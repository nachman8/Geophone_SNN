# Human Stepping File Processor

This tool processes a 60-second geophone file containing human stepping activity and generates resonator-based spikegrams for analysis.

## File Structure Expected

Your 60-second geophone file should contain:
- **32.5 seconds of human stepping** (0-32.5s)
- **27.5 seconds of background/no stepping** (32.5-60s)
- **1000 Hz sampling frequency**
- **CSV format** with either an 'amplitude' column or signal data in the second column

## Quick Start

### Option 1: Simple Command Line

```bash
cd project/MyCode
python run_human_processor.py /path/to/your/geophone_file.csv
```

### Option 2: Edit and Run

1. Edit `run_human_processor.py`
2. Update the `file_path` variable with your actual file path
3. Run: `python run_human_processor.py`

### Option 3: Direct Processing

```bash
python process_human_file_simple.py /path/to/your/geophone_file.csv
```

## What the Script Does

1. **Loads and validates** your 60-second CSV file
2. **Splits the signal** at 32.5 seconds:
   - Human stepping section (0-32.5s) → `human_new/`
   - Background section (32.5-60s) → `human_nothing_new/`
3. **Processes each section** using resonator grids:
   - Human-optimized resonators for stepping section
   - Background-optimized resonators for quiet section
4. **Generates spikegrams** using parallel processing
5. **Creates visualizations** showing raw signal and spikegrams
6. **Saves results** in organized directory structure

## Output Structure

```
chunked_output/
├── human_new/                          # Human stepping chunks
│   ├── chunk_0/
│   │   ├── chunk_0_data.pkl           # Spikegram data
│   │   └── human_new_chunk_0_visualization.png
│   └── chunk_1/ (if applicable)
├── human_nothing_new/                  # Background chunks  
│   ├── chunk_0/
│   │   ├── chunk_0_data.pkl           # Background spikegram data
│   │   └── human_nothing_new_chunk_0_visualization.png
│   └── chunk_1/ (if applicable)
└── processing_summary.pkl              # Overall processing summary
```

## Configuration Options

You can modify these parameters in the script:

- `human_end_time=32.5`: Time when human stepping ends (seconds)
- `chunk_duration=30`: Size of processing chunks (seconds)
- `num_processes=15`: Number of parallel processes (adjust based on your CPU)

## Requirements

- Python 3.x
- Required packages: numpy, pandas, scipy, matplotlib, joblib
- SCTN library (already configured in the script)
- Sufficient disk space for output files
- At least 4-8GB RAM for resonator processing

## Troubleshooting

### Common Issues

1. **File not found error**
   - Check the file path is correct
   - Use absolute paths if relative paths don't work

2. **Memory errors**
   - Reduce `num_processes` from 15 to 8 or lower
   - Check available RAM

3. **CSV format issues**
   - Ensure your file has proper CSV format
   - Check that signal data is in 'amplitude' column or second column

4. **Processing takes too long**
   - This is normal - resonator processing is computationally intensive
   - You'll see progress bars showing completion percentage

### Performance Tips

- Use SSD storage for faster I/O
- Close other applications to free up RAM
- Adjust `num_processes` based on your CPU cores
- Monitor CPU and memory usage during processing

## Data Format Details

### Input CSV File
```csv
time,amplitude
0.000,0.123
0.001,0.456
0.002,-0.234
...
```

### Output Pickle Files
Each `chunk_X_data.pkl` contains:
- `chunk_idx`: Chunk number
- `start_time`: Start time in original file
- `duration`: Chunk duration
- `signal`: Raw signal data
- `resonator_outputs`: Raw resonator spike data
- `max_spikes_spectrogram`: Full frequency spikegram
- `spikes_bands_spectrogram`: Band-grouped spikegram (8 frequency bands)
- `all_freqs`: Resonator frequencies used

## Integration with Main System

The output directories `human_new/` and `human_nothing_new/` are designed to work with the existing ensemble SNN training system. You can use these directories directly with the feature extraction and classification pipeline in `Geophone_Model_SNN.py`.

## Example Usage

```bash
# Navigate to the directory
cd /home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode

# Process your file
python run_human_processor.py /home/user/data/geophone_60sec.csv

# Check the results
ls -la chunked_output/human_new/
ls -la chunked_output/human_nothing_new/
```

## Next Steps

After processing:
1. **Verify the visualizations** to ensure proper signal processing
2. **Use the generated directories** with the ensemble SNN training system
3. **Analyze the spikegrams** for human stepping patterns vs background noise
4. **Train classification models** using the processed features 