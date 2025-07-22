# Comprehensive Geophone Classification Analysis
## Following EEG Study Methodology: FFT vs SCTN Feature Comparison

### Executive Summary
This analysis implements a comprehensive geophone classification system following the methodology from "SCTN-based EEG classification" research, comparing FFT-based features against SCTN-based features across multiple machine learning models for car and human detection tasks.

---

## 📊 Dataset Overview

**Total Dataset:** 32 chunks from geophone recordings
- **Car Detection Task:** 11 samples (7 car, 4 car_nothing)  
- **Human Detection Task:** 21 samples (12 human, 9 human_nothing)
- **Source:** Processed spikegram data from resonator-based frequency analysis

---

## 🔬 Feature Extraction Methods

### 1. FFT-Based Features (75 features)
Following EEG study frame-based analysis:
- **Frame Durations:** 5, 10, 15, 20 ms (variable optimization)
- **Frequency Bands:** 8 geophone-specific bands (20-100 Hz)
- **Feature Types:** 
  - Frame-based FFT power analysis
  - Statistical features (mean, std, max, min, median, IQR)
  - Global spectral characteristics (centroid, spread)
  - Band energy distribution

### 2. SCTN-Based Features (35 features)
Enhanced pattern-based analysis:
- **Signal-Specific Patterns:** Car periodicity, human burstiness
- **Frequency Band Analysis:** Car (30-48 Hz), Human (60-85 Hz)
- **Feature Types:**
  - Signature energy ratios
  - Temporal pattern analysis
  - Cross-band correlations
  - Signal quality metrics

---

## 🤖 Classification Models Evaluated

Following EEG study model selection:
1. **SVM** (RBF kernel)
2. **Random Forest** (100 estimators)
3. **XGBoost** (Gradient boosting)
4. **LightGBM** (Light gradient boosting)
5. **KNN** (5 neighbors)
6. **Weighted Ensemble** (SVM:2 + RF:3 + XGB:1)

---

## 📈 Results Summary

### Individual Model Performance

| Model | Task | FFT | SCTN | Improvement |
|-------|------|-----|------|-------------|
| SVM | Car Detection | 90.0% | 83.3% | -6.7% |
| Random Forest | Car Detection | **100.0%** | 90.0% | -10.0% |
| XGBoost | Car Detection | 63.3% | 63.3% | +0.0% |
| LightGBM | Car Detection | 63.3% | 63.3% | +0.0% |
| KNN | Car Detection | 63.3% | 63.3% | +0.0% |
| SVM | Human Detection | 85.0% | 85.0% | +0.0% |
| Random Forest | Human Detection | 85.0% | 81.0% | -4.0% |
| XGBoost | Human Detection | **86.0%** | 86.0% | +0.0% |
| LightGBM | Human Detection | 57.0% | 57.0% | +0.0% |
| KNN | Human Detection | 66.0% | 72.0% | **+6.0%** |

### Weighted Ensemble Performance
| Task | FFT Ensemble | SCTN Ensemble | Improvement |
|------|--------------|---------------|-------------|
| Car Detection | **91.7%** | 83.3% | -8.3% |
| Human Detection | 85.7% | 85.7% | +0.0% |

---

## 📊 Statistical Analysis

### Overall Performance Statistics
- **Average FFT Performance:** 78.0%
- **Average SCTN Performance:** 76.1%
- **Average Improvement:** -1.9%
- **Standard Deviation:** 4.3%
- **Max Improvement:** +6.0% (KNN on human detection)
- **Min Improvement:** -10.0% (Random Forest on car detection)
- **SCTN Superior Cases:** 1/12 (8.3%)

### Key Findings
1. **FFT Dominance:** FFT features outperformed SCTN features in most cases
2. **Task Dependency:** Performance varied significantly by detection task
3. **Model Sensitivity:** Different models showed varying sensitivity to feature types
4. **Best Individual Performance:** Random Forest with FFT (100% car detection)

---

## 🔍 Detailed Feature Analysis

### Most Important Features by Task

**Car Detection (FFT):**
1. Feature 3: 0.081 (spectral characteristics)
2. Feature 72: 0.079 (band energy distribution)
3. Feature 15: 0.074 (frame-based power)

**Car Detection (SCTN):**
1. Feature 0: 0.104 (car signature ratio)
2. Feature 16: 0.101 (individual band features)
3. Feature 22: 0.096 (band analysis)

**Human Detection (FFT):**
1. Feature 50: 0.135 (high-frequency characteristics)
2. Feature 69: 0.099 (spectral distribution)

**Human Detection (SCTN):**
1. Feature 4: 0.119 (signal-to-noise ratio)
2. Feature 28: 0.116 (individual band features)
3. Feature 1: 0.106 (human signature ratio)

---

## 📋 Comparison with EEG Study Results

### EEG Study Baseline Performance
| Model | Frame Duration | FFT | SCTN | Improvement |
|-------|---------------|-----|------|-------------|
| Weighted Ensemble | 10ms | 90.1% | **93.9%** | +3.8% |
| SVM | 20ms | 91.5% | **93.2%** | +1.7% |
| XGBoost | 10ms | 88.8% | **92.6%** | +3.8% |
| Random Forest | 5ms | 86.8% | **92.5%** | +5.7% |
| LightGBM | 15ms | 89.4% | **92.0%** | +2.6% |
| KNN | 20ms | 63.8% | **69.0%** | +5.2% |

### Geophone Study Results
**Average Improvement:** -1.9% (SCTN underperformed)

**Key Differences:**
1. **Opposite Trend:** Unlike EEG where SCTN consistently outperformed FFT, geophone data favored FFT
2. **Signal Characteristics:** EEG neural patterns may be more suited to SCTN analysis than mechanical vibrations
3. **Feature Engineering:** Geophone-specific patterns may require different SCTN feature designs

---

## 🎯 Task-Specific Performance Analysis

### Car Detection
- **Best FFT:** Random Forest (100.0%)
- **Best SCTN:** Random Forest (90.0%)
- **Ensemble FFT:** 91.7%
- **Ensemble SCTN:** 83.3%
- **Analysis:** Strong FFT advantage, possibly due to periodic car signatures being well-captured by frequency domain analysis

### Human Detection  
- **Best FFT:** XGBoost (86.0%)
- **Best SCTN:** XGBoost (86.0%)
- **Ensemble FFT:** 85.7%
- **Ensemble SCTN:** 85.7%
- **Analysis:** Comparable performance, with slight SCTN advantage in KNN model (+6.0%)

---

## 🔬 Technical Implementation Details

### System Architecture
```
📁 comprehensive_geophone_classifier.py
├── FFTFeatureExtractor (75 features)
├── SCTNFeatureExtractor (35 features)  
├── GeophoneClassificationSystem
└── Multi-model evaluation pipeline

📁 enhanced_ensemble_classifier.py
├── EnhancedEnsembleAnalyzer
├── Weighted ensemble evaluation
├── Feature importance analysis
└── Comprehensive reporting
```

### Cross-Validation Strategy
- **Method:** 5-fold Stratified K-Fold (reduced to 3-fold for small datasets)
- **Scoring:** Accuracy
- **Random State:** 42 (reproducible results)

---

## 🔍 Conclusions

### Primary Findings
1. **FFT Superiority:** FFT features outperformed SCTN features for geophone classification (-1.9% average improvement)
2. **Task Dependence:** Car detection showed stronger FFT advantage than human detection
3. **Model Variance:** Different models responded differently to feature types
4. **Dataset Limitation:** Small sample size (32 chunks) limits statistical significance

### Methodological Insights
1. **Domain Specificity:** SCTN advantages observed in EEG may not transfer to mechanical vibration analysis
2. **Feature Engineering:** Geophone-specific SCTN features may need refinement
3. **Signal Characteristics:** FFT may be naturally suited to periodic mechanical signatures

### Comparison with EEG Study
- **EEG Results:** SCTN consistently outperformed FFT (average +3.9% improvement)
- **Geophone Results:** FFT outperformed SCTN (average -1.9% improvement)
- **Domain Difference:** Neural signal patterns vs mechanical vibration patterns

---

## 💡 Recommendations

### For Geophone Classification
1. **Feature Fusion:** Combine FFT and SCTN features for hybrid approach
2. **Dataset Expansion:** Collect more samples for robust statistical analysis
3. **SCTN Refinement:** Develop geophone-specific SCTN feature engineering
4. **Temporal Analysis:** Explore longer time-scale SCTN patterns

### For Future Research
1. **Cross-Domain Analysis:** Study SCTN effectiveness across different signal types
2. **Hybrid Approaches:** Investigate FFT-SCTN feature fusion methods
3. **Real-time Implementation:** Optimize best-performing models for deployment

---

## 📂 Implementation Files

### Core Analysis Files
- `comprehensive_geophone_classifier.py` - Main classification system
- `enhanced_ensemble_classifier.py` - Weighted ensemble analysis
- `old_archive/` - Previous implementations and iterations

### Supporting Data
- `chunked_output/` - Processed geophone chunk data
- Analysis follows established spikegram processing pipeline

---

## 🎉 Final Assessment

**Following EEG Study Methodology Success:** ✅ Complete implementation with comparable analysis depth

**Key Achievement:** Comprehensive feature comparison framework that can be applied to other signal classification domains

**Primary Insight:** Domain-specific signal characteristics significantly influence the effectiveness of different feature extraction approaches, highlighting the importance of tailored feature engineering for each application domain.

---

*Analysis completed following rigorous EEG study methodology with 5 classification models, weighted ensemble evaluation, and comprehensive statistical analysis.* 