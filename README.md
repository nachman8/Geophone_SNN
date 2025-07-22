# Geophone_SNN

> Geophone Signal Processing with Resonator-Based Analysis
This project focuses on processing seismic signals from geophones using resonator-based signal analysis techniques. The system is designed to detect and classify different types of activities (human walking, vehicle movements) through their seismic signatures.

Key Components:
Resonator-Based Processing: Uses a grid of tuned resonators at specific frequencies to analyze seismic signals
Multi-sensor Support: Can process data from multiple geophone sensors simultaneously
Frequency Band Analysis: Categorizes signals into meaningful frequency bands (e.g., 20-30Hz, 34-40Hz for car movements, 60-70Hz for human activity)
Memory-Efficient Processing: Handles large files by processing them in chunks
Visualization Tools: Generates spectrograms and resonator output visualizations for analysis
Integration with SCTN Framework:
The project integrates with the Spiking Cellular Temporal Neural (SCTN) framework, which appears to be a specialized neural network framework for processing temporal signals, likely using spiking neural network concepts.

Signal Processing Pipeline:
Signal normalization and preprocessing
Parallel processing through resonator grids optimized for different signal types
Conversion of resonator outputs to spike-based spectrograms
Frequency band analysis for classification
The code is particularly focused on distinguishing between human footsteps and vehicle movements based on their distinct frequency patterns in seismic data, with resonator grids specifically tuned for each type of signal.

![License](https://img.shields.io/badge/license-MIT-green) ![Version](https://img.shields.io/badge/version-1.0.0-blue) ![Language](https://img.shields.io/badge/language-Python-yellow) ![GitHub](https://img.shields.io/badge/GitHub-nachman8/Geophone_SNN-black?logo=github) ![Build Status](https://img.shields.io/github/actions/workflow/status/nachman8/Geophone_SNN/ci.yml?branch=main)

## ℹ️ Project Information

- **👤 Author:** Nachman Mimoun
- **📦 Version:** 1.0.0
- **📄 License:** MIT
- **📂 Repository:** [https://github.com/nachman8/Geophone_SNN](https://github.com/nachman8/Geophone_SNN)

