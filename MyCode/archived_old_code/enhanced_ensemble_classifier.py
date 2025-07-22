#!/usr/bin/env python3
"""
Enhanced Ensemble Geophone Classifier
Adding Weighted Ensemble approach from EEG study + detailed analysis
"""

import numpy as np
import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Import the comprehensive system
sys.path.insert(0, "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode")
from comprehensive_geophone_classifier import GeophoneClassificationSystem

class EnhancedEnsembleAnalyzer:
    """Enhanced analysis with weighted ensemble following EEG study"""
    
    def __init__(self):
        self.base_system = GeophoneClassificationSystem()
        
        # Weighted ensemble (best performer in EEG study)
        self.weighted_ensemble = VotingClassifier(
            estimators=[
                ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
            ],
            voting='soft',
            weights=[2, 3, 1]  # Weight based on typical performance
        )
        
        print("🔬 Enhanced Ensemble Analyzer initialized")
        print("   🏆 Weighted ensemble: SVM(2) + RF(3) + XGB(1)")
    
    def evaluate_ensemble_performance(self, datasets):
        """Evaluate weighted ensemble performance like EEG study"""
        print(f"\n🏆 WEIGHTED ENSEMBLE EVALUATION")
        print("=" * 50)
        
        ensemble_results = {}
        
        for task_name, dataset in datasets.items():
            print(f"\n📊 TASK: {task_name.upper()}")
            print("-" * 40)
            
            X_fft = dataset['fft_features']
            X_sctn = dataset['sctn_features']
            y = dataset['labels']
            
            # FFT Ensemble
            fft_ensemble_score = self._evaluate_ensemble(X_fft, y, "FFT")
            
            # SCTN Ensemble
            sctn_ensemble_score = self._evaluate_ensemble(X_sctn, y, "SCTN")
            
            improvement = sctn_ensemble_score - fft_ensemble_score
            
            ensemble_results[task_name] = {
                'FFT_Ensemble': fft_ensemble_score,
                'SCTN_Ensemble': sctn_ensemble_score,
                'Improvement': improvement
            }
            
            print(f"Weighted Ensemble FFT:  {fft_ensemble_score:.1%}")
            print(f"Weighted Ensemble SCTN: {sctn_ensemble_score:.1%}")
            print(f"Improvement:            {improvement:+.1%}")
        
        return ensemble_results
    
    def _evaluate_ensemble(self, X, y, feature_type):
        """Evaluate ensemble with cross-validation"""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for small dataset
            scores = cross_val_score(self.weighted_ensemble, X_scaled, y, cv=cv, scoring='accuracy')
            
            return np.mean(scores)
            
        except Exception as e:
            print(f"⚠️ Error in {feature_type} ensemble: {e}")
            return 0.0
    
    def detailed_feature_analysis(self, datasets):
        """Analyze which features contribute most to performance"""
        print(f"\n🔬 DETAILED FEATURE ANALYSIS")
        print("=" * 50)
        
        for task_name, dataset in datasets.items():
            print(f"\n📊 ANALYSIS: {task_name.upper()}")
            print("-" * 40)
            
            X_fft = dataset['fft_features']
            X_sctn = dataset['sctn_features']
            y = dataset['labels']
            
            # Feature importance analysis using Random Forest
            self._analyze_feature_importance(X_fft, y, "FFT", task_name)
            self._analyze_feature_importance(X_sctn, y, "SCTN", task_name)
    
    def _analyze_feature_importance(self, X, y, feature_type, task_name):
        """Analyze feature importance"""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            # Get top 5 most important features
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            
            print(f"{feature_type} Top Features for {task_name}:")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Feature {idx}: {importances[idx]:.3f}")
            
        except Exception as e:
            print(f"⚠️ Error analyzing {feature_type} features: {e}")
    
    def generate_eeg_comparison_table(self, ensemble_results, individual_results):
        """Generate comparison table similar to EEG study Table 1"""
        print(f"\n📊 EEG STUDY COMPARISON TABLE")
        print("=" * 60)
        print("Model               Frame Duration  FFT      SCTN")
        print("-" * 60)
        
        # Individual model results
        print("SVM                 20ms           91.5%    93.2%")  # Example from EEG study
        print("Random Forest       5ms            86.8%    92.5%")  # Example from EEG study
        print("XGBoost            10ms           88.8%    92.6%")   # Example from EEG study
        print("LightGBM           15ms           89.4%    92.0%")   # Example from EEG study
        print("KNN                20ms           63.8%    69.0%")   # Example from EEG study
        
        # Our results
        print("\n--- OUR GEOPHONE RESULTS ---")
        for task_name, results in individual_results.items():
            print(f"\n{task_name.upper()}:")
            for model_name, model_results in results.items():
                fft_acc = model_results['FFT']
                sctn_acc = model_results['SCTN']
                print(f"{model_name:<15}    Variable       {fft_acc:.1%}    {sctn_acc:.1%}")
        
        # Ensemble results
        print(f"\n--- WEIGHTED ENSEMBLE RESULTS ---")
        for task_name, results in ensemble_results.items():
            fft_ens = results['FFT_Ensemble']
            sctn_ens = results['SCTN_Ensemble']
            improvement = results['Improvement']
            print(f"Weighted Ensemble   {task_name:<10} {fft_ens:.1%}    {sctn_ens:.1%}    ({improvement:+.1%})")
    
    def comprehensive_evaluation_report(self, ensemble_results, individual_results):
        """Generate comprehensive evaluation report"""
        print(f"\n📋 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)
        
        # Calculate overall statistics
        all_fft = []
        all_sctn = []
        all_improvements = []
        
        for task_results in individual_results.values():
            for model_results in task_results.values():
                all_fft.append(model_results['FFT'])
                all_sctn.append(model_results['SCTN'])
                all_improvements.append(model_results['Improvement'])
        
        # Add ensemble results
        for ens_results in ensemble_results.values():
            all_fft.append(ens_results['FFT_Ensemble'])
            all_sctn.append(ens_results['SCTN_Ensemble'])
            all_improvements.append(ens_results['Improvement'])
        
        print(f"\n📈 OVERALL PERFORMANCE STATISTICS")
        print(f"Average FFT Performance:     {np.mean(all_fft):.1%}")
        print(f"Average SCTN Performance:    {np.mean(all_sctn):.1%}")
        print(f"Average Improvement:         {np.mean(all_improvements):+.1%}")
        print(f"Std Dev of Improvement:      {np.std(all_improvements):.1%}")
        print(f"Max Improvement:             {np.max(all_improvements):+.1%}")
        print(f"Min Improvement:             {np.min(all_improvements):+.1%}")
        print(f"SCTN Superior in:            {np.sum(np.array(all_improvements) > 0)}/{len(all_improvements)} cases")
        
        print(f"\n🎯 TASK-SPECIFIC PERFORMANCE")
        for task_name, ens_results in ensemble_results.items():
            best_fft = np.max([results['FFT'] for results in individual_results[task_name].values()])
            best_sctn = np.max([results['SCTN'] for results in individual_results[task_name].values()])
            ens_fft = ens_results['FFT_Ensemble']
            ens_sctn = ens_results['SCTN_Ensemble']
            
            print(f"\n{task_name.upper()}:")
            print(f"  Best Individual FFT:     {best_fft:.1%}")
            print(f"  Best Individual SCTN:    {best_sctn:.1%}")
            print(f"  Ensemble FFT:            {ens_fft:.1%}")
            print(f"  Ensemble SCTN:           {ens_sctn:.1%}")
            print(f"  Ensemble vs Best FFT:    {ens_fft - best_fft:+.1%}")
            print(f"  Ensemble vs Best SCTN:   {ens_sctn - best_sctn:+.1%}")
        
        # Conclusions
        print(f"\n🔍 CONCLUSIONS")
        print("-" * 30)
        
        if np.mean(all_improvements) > 0.01:  # >1% improvement
            print("✅ SCTN features show clear advantage over FFT features")
        elif np.mean(all_improvements) > -0.01:  # Within 1%
            print("📊 SCTN and FFT features show comparable performance")
        else:
            print("⚠️ FFT features outperform SCTN features in this dataset")
        
        best_ensemble_improvement = np.max([r['Improvement'] for r in ensemble_results.values()])
        if best_ensemble_improvement > 0.05:  # >5% improvement
            print("🏆 Weighted ensemble shows significant SCTN advantage")
        elif best_ensemble_improvement > 0:
            print("📈 Weighted ensemble shows moderate SCTN advantage")
        else:
            print("📉 Weighted ensemble favors FFT features")
        
        # Dataset limitations
        total_samples = 32  # We know from the previous output
        if total_samples < 50:
            print("⚠️ Limited dataset size may affect statistical significance")
        
        return {
            'avg_fft': np.mean(all_fft),
            'avg_sctn': np.mean(all_sctn),
            'avg_improvement': np.mean(all_improvements),
            'best_ensemble_improvement': best_ensemble_improvement,
            'total_samples': total_samples
        }


def main():
    """Enhanced main function with ensemble analysis"""
    print("🚀 ENHANCED ENSEMBLE GEOPHONE CLASSIFICATION")
    print("=" * 60)
    print("Following EEG Study Methodology with Weighted Ensemble")
    print()
    
    # Initialize enhanced analyzer
    analyzer = EnhancedEnsembleAnalyzer()
    
    # Load and process data using base system
    chunks_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
    chunk_data = analyzer.base_system.load_chunk_data(chunks_dir)
    
    if not chunk_data:
        print("❌ No chunk data found")
        return
    
    # Extract features
    features_fft, features_sctn, labels, chunk_info = analyzer.base_system.extract_features_from_chunks(
        chunk_data, feature_type='both'
    )
    
    # Create task-specific datasets
    datasets = analyzer.base_system.create_task_specific_datasets(
        features_fft, features_sctn, labels, chunk_info
    )
    
    # Evaluate individual models
    individual_results = analyzer.base_system.evaluate_models(datasets)
    
    # Evaluate ensemble performance
    ensemble_results = analyzer.evaluate_ensemble_performance(datasets)
    
    # Detailed feature analysis
    analyzer.detailed_feature_analysis(datasets)
    
    # Generate EEG comparison table
    analyzer.generate_eeg_comparison_table(ensemble_results, individual_results)
    
    # Comprehensive evaluation report
    final_stats = analyzer.comprehensive_evaluation_report(ensemble_results, individual_results)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"Overall SCTN vs FFT: {final_stats['avg_improvement']:+.1%} average improvement")
    
    return final_stats, ensemble_results, individual_results


if __name__ == "__main__":
    results = main() 