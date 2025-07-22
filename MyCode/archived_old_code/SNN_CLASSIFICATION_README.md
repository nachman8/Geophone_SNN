# Spiking Neural Network Classification for Geophone Signals

## Overview

This system implements a Spiking Neural Network (SNN) for classifying geophone signals into three categories:
- **Car**: Vehicle activity detected
- **Human**: Human activity (footsteps) detected  
- **Nothing**: No significant activity

## Architecture

### Network Structure
```
Input Layer (8-32 neurons) → Hidden Layer (30-50 neurons) → Output Layer (3 neurons)
```

- **Input Layer**: Receives spike-encoded resonator outputs from 8 frequency bands
- **Hidden Layer**: SCTN neurons with STDP learning for feature extraction
- **Output Layer**: 3 binary neurons (car/human/nothing) with supervised STDP

### Key Features

1. **Resonator-based Preprocessing**: Uses trained resonators to extract frequency-specific features
2. **Spike Encoding**: Converts analog features to biologically-inspired spike trains
3. **STDP Learning**: Unsupervised learning in hidden layer + supervised learning in output layer
4. **Spectral Analysis**: Intelligent segmentation based on frequency band activity

## Implementation Details

### Spike Encoding Methods

1. **Rate Coding**: Higher feature values → higher spike rates (0-100 Hz)
2. **Temporal Coding**: Higher feature values → earlier spike times
3. **Enhanced Encoding**: Includes jitter and realistic spike timing

### Learning Rules

- **Hidden Layer**: Spike-Timing Dependent Plasticity (STDP)
  - A_LTP = 0.01 (potentiation)
  - A_LTD = -0.005 (depression)
  - τ = 20ms (time constant)

- **Output Layer**: Supervised STDP
  - Target-driven weight updates
  - Class-specific desired outputs

### Spectral Analysis

The system analyzes spectrograms to identify segments with characteristic patterns:

- **Car Detection**: Looks for periodic patterns in 30-50 Hz range
- **Human Detection**: Looks for burst patterns in 60-85 Hz range
- **Nothing Detection**: Identifies low-activity segments

## Usage

### Basic Usage

```python
from snn_classification import run_enhanced_car_classification

# Run car vs nothing classification
results = run_enhanced_car_classification()

if results:
    snn, accuracy, report, history = results
    print(f"Test Accuracy: {accuracy:.3f}")
```

### Custom Training

```python
from snn_classification import GeophoneSNN

# Create SNN
snn = GeophoneSNN(n_hidden=40, learning_rate=0.015)

# Prepare your data (X = features, y = labels)
X_train, X_test, y_train, y_test = prepare_your_data()

# Train
training_history = snn.train(X_train, y_train, n_epochs=80)

# Evaluate
accuracy, report, cm = snn.evaluate(X_test, y_test)

# Save model
snn.save_model("my_snn_model.pkl")
```

## Performance Optimization

### Data Segmentation
- Segments data into 10-second windows with 50% overlap
- Analyzes frequency band activity to identify relevant segments
- Only trains on segments with characteristic signal patterns

### Network Parameters
- **Spike Duration**: 200ms for detailed temporal encoding
- **Hidden Neurons**: 30-50 for good feature extraction
- **Training Epochs**: 80-100 for convergence
- **Learning Rate**: 0.01-0.02 for stable learning

## Expected Results

### Car vs Nothing Classification
- **Accuracy**: 80-95% on test set
- **Precision**: High for both car and nothing detection
- **Training Time**: 5-10 minutes on modern hardware

### Feature Importance
The system automatically focuses on:
1. **Car Signals**: 30-48 Hz frequency bands (engine/vibration)
2. **Human Signals**: 60-85 Hz frequency bands (footstep impacts)
3. **Temporal Patterns**: Periodic vs. burst activity

## Files Structure

```
project/MyCode/
├── snn_classification.py          # Main SNN implementation
├── resonator_work.py             # Resonator processing functions
├── SNN_CLASSIFICATION_README.md  # This file
└── output_plots/                 # Generated visualizations
    ├── snn_training_curve.png
    └── snn_classification_results.png
```

## Dependencies

```python
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.5.0
scikit-learn >= 1.0.0
```

Plus the sctnN library for SNN implementation.

## Future Extensions

### Human vs Human_Nothing
The same approach can be extended to human classification by:
1. Using human-optimized resonator grids
2. Adjusting frequency band importance weights
3. Modifying threshold parameters for human activity detection

### Three-Class Classification
Full car/human/nothing classification by:
1. Combining all datasets
2. Using balanced sampling
3. Training the 3-output network simultaneously

## Troubleshooting

### Common Issues

1. **Low Accuracy**: Check data quality and segment selection
2. **Training Instability**: Reduce learning rate or adjust network size
3. **Memory Issues**: Reduce spike duration or batch size
4. **Slow Training**: Use fewer hidden neurons or shorter spike trains

### Debug Tips

- Monitor training accuracy curve for overfitting
- Check confusion matrix for class-specific issues
- Visualize spike trains to verify encoding
- Analyze feature distributions across classes

## Citation

If you use this SNN classification system, please cite:

```
Geophone Signal Classification using Spiking Neural Networks
Resonator-based Feature Extraction with STDP Learning
Project_Geo SNN Implementation, 2024
``` 