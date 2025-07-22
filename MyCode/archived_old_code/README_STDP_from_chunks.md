# STDP Classification from Existing Chunks

This directory contains scripts to load your existing processed chunks and create STDP classification datasets without re-processing the resonators.

## Overview

Since you already have processed chunks in `chunked_output_30s/`, this approach:
1. ✅ **Loads existing chunks** (no re-processing needed)
2. ✅ **Creates SNN datasets** with temporal continuity preserved
3. ✅ **Prepares STDP networks** compatible with old notebooks
4. ✅ **Saves datasets** in old notebook format for easy use

## Files Created

1. **`load_chunks_and_classify.py`** - Main script that loads chunks and creates STDP datasets
2. **`run_stdp_from_chunks.py`** - Simple runner script
3. **`README_STDP_from_chunks.md`** - This documentation

## Prerequisites

- ✅ Existing chunks in `chunked_output_30s/` (already done)
- ✅ `resonator_work.py` in the same directory
- ✅ All dependencies from resonator processing

## How to Run

### Option 1: Simple Runner (Recommended)
```bash
cd project/MyCode
python run_stdp_from_chunks.py
```

### Option 2: Direct Import
```bash
cd project/MyCode
python -c "from load_chunks_and_classify import run_stdp_classification_from_chunks; run_stdp_classification_from_chunks()"
```

### Option 3: Interactive Script
```bash
cd project/MyCode
python load_chunks_and_classify.py
```

## What It Does

### Step 1: Load Existing Chunks
- Automatically discovers chunks in `chunked_output_30s/`
- Loads chunk indices for both car and human data
- Validates chunk integrity

### Step 2: Create SNN Datasets
- Combines chunks while preserving temporal continuity
- Creates spike trains compatible with old notebooks
- Saves datasets in multiple formats

### Step 3: Prepare STDP Training
- Sets up STDP network structure
- Prepares training data in old notebook format
- Creates frequency-specific spike trains

### Step 4: Save for Training
- Saves datasets as numpy arrays (like old notebooks)
- Creates metadata and summary files
- Provides direct access to spike trains

## Output Structure

After running, you'll have:

```
project/MyCode/
├── snn_dataset_car_combined_from_chunks/
│   ├── spikes_22.1Hz.npy
│   ├── spikes_30.5Hz.npy
│   ├── events_22.1Hz.npy
│   ├── dataset_metadata.pkl
│   └── dataset_summary.json
├── snn_dataset_human_combined_from_chunks/
│   ├── spikes_22.1Hz.npy
│   ├── spikes_76.3Hz.npy
│   ├── events_22.1Hz.npy
│   ├── dataset_metadata.pkl
│   └── dataset_summary.json
└── stdp_training_summary.json
```

## Using the Data (Old Notebook Style)

```python
import numpy as np

# Load car data (example)
car_spikes_22_1 = np.load('snn_dataset_car_combined_from_chunks/spikes_22.1Hz.npy')
car_spikes_30_5 = np.load('snn_dataset_car_combined_from_chunks/spikes_30.5Hz.npy')

# Load human data (example)
human_spikes_22_1 = np.load('snn_dataset_human_combined_from_chunks/spikes_22.1Hz.npy')
human_spikes_76_3 = np.load('snn_dataset_human_combined_from_chunks/spikes_76.3Hz.npy')

# Use in STDP training (compatible with old notebooks)
network.input_full_data_spikes(car_spikes_22_1)
network.input_full_data_spikes(human_spikes_22_1)
```

## Key Advantages

### ⚡ Speed
- **No re-processing** - Uses existing chunks
- **Direct loading** - Fast dataset creation
- **Immediate use** - Ready for STDP training

### 🧠 Compatibility
- **Old notebook format** - Same as your previous work
- **STDP ready** - Temporal continuity preserved
- **Easy integration** - Drop-in replacement

### 📊 Comprehensive
- **All frequencies** - Complete resonator output
- **Full datasets** - Car and human data
- **Metadata included** - Easy to understand and use

## Workflow Comparison

### ❌ Previous Approach
```
Raw Data → Resonator Processing → Chunks → SNN Datasets → STDP Training
    ↑                                           ↑
Takes hours/days                          Start here now!
```

### ✅ New Approach (This Script)
```
Existing Chunks → SNN Datasets → STDP Training
       ↑              ↑             ↑
   Already done    Fast (minutes)  Ready!
```

## Troubleshooting

### No chunks found
- Check that `chunked_output_30s/` exists
- Verify chunks were processed correctly
- Look for `chunk_index.pkl` files

### Import errors
- Make sure `resonator_work.py` is in the same directory
- Check that all dependencies are installed

### Memory issues
- The script loads chunks efficiently
- Should handle large datasets without problems

### Dataset format questions
- Uses exact same format as old notebooks
- Compatible with all existing STDP code
- Numpy arrays for each frequency

## Next Steps

1. **Run the script** to create STDP datasets
2. **Load the datasets** in your STDP training code
3. **Use existing methods** - fully compatible with old notebooks
4. **Train STDP networks** on car vs human classification

## Technical Details

### Temporal Continuity
- Spike timing is preserved across chunks
- File boundaries are tracked
- STDP learning works correctly

### Data Format
- Binary spike trains (0/1 arrays)
- Event timestamps for efficiency
- Metadata for easy loading

### Frequency Coverage
- All resonator frequencies included
- Car-optimized and human-optimized grids
- Common frequencies identified for training

---

## Summary

This approach gives you the best of both worlds:
- ✅ **Efficient**: Uses existing processed chunks
- ✅ **Compatible**: Works with all your old notebook code
- ✅ **Complete**: Full STDP-ready datasets
- ✅ **Fast**: Ready for training in minutes, not hours

Perfect for continuing your STDP research without re-processing!
