# Human Chunk Processing

This directory contains scripts to process human data files using the same chunked approach as your existing car data.

## Files Created

1. **`process_human_chunks.py`** - Main processing script (imports functions from resonator_work.py)
2. **`run_human_processing.py`** - Simple runner script to execute the processing
3. **`README_human_processing.md`** - This file

## What This Does

- Processes `human.csv` and `human_nothing.csv` using the same chunked approach as car data
- Uses human-optimized resonator grid from `resonator_work.py`
- Saves chunks to `chunked_output_30s/human/` and `chunked_output_30s/human_nothing/`
- Creates SNN-compatible datasets for STDP training
- Memory efficient - processes files in 30-second chunks

## How to Run

### Option 1: Simple Runner (Recommended)
```bash
cd project/MyCode
python run_human_processing.py
```

### Option 2: Direct Import
```bash
cd project/MyCode
python -c "from process_human_chunks import process_human_files_only; process_human_files_only()"
```

### Option 3: Run as Script
```bash
cd project/MyCode
python process_human_chunks.py
```

## Requirements

- `resonator_work.py` must be in the same directory
- Human data files must exist at `~/data/human.csv` and `~/data/human_nothing.csv`
- All dependencies from `resonator_work.py` must be available

## Output Structure

After running, you'll have:

```
chunked_output_30s/
├── car/                    # Your existing car chunks
├── car_nothing/            # Your existing car_nothing chunks
├── human/                  # NEW: human chunks
│   ├── chunk_0/
│   ├── chunk_1/
│   └── ...
├── human_nothing/          # NEW: human_nothing chunks
│   ├── chunk_0/
│   ├── chunk_1/
│   └── ...
└── snn_dataset_human_combined/  # SNN training dataset
```

## Next Steps

1. **Combine with Car Data**: Use existing functions to combine car and human chunks
2. **SNN Training**: Use the SNN datasets for STDP training
3. **Analysis**: Both car and human data now available in chunked format

## Troubleshooting

- **Import errors**: Make sure `resonator_work.py` is in the same directory
- **File not found**: Check that human data files exist at `~/data/`
- **Memory issues**: The chunked approach should handle large files efficiently
- **Permission errors**: Make sure you have write access to the current directory

## Features

- ✅ Memory efficient chunked processing
- ✅ Human-optimized resonator grid
- ✅ SNN STDP compatibility
- ✅ Temporal continuity preserved
- ✅ Compatible with existing car chunks
- ✅ Same processing pipeline as car data
