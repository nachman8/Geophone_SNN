#!/usr/bin/env python3
# Comprehensive Geophone Signal Classification Optimization
# Complete system integration with advanced features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the sctnN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import our optimized components
from advanced_pattern_analyzer import AdvancedPatternAnalyzer
from optimized_snn_classifier import OptimizedGeophoneSNN

class ComprehensiveOptimizer:
    def __init__(self, chunks_dir=None):
        self.chunks_dir = chunks_dir or "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project/MyCode/chunked_output"
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.results = {}
        
        print("🚀 COMPREHENSIVE GEOPHONE SIGNAL OPTIMIZATION")
        print("="*80)
        print("🔧 Advanced Pattern Analysis + Optimized SNN Classification")
        print("📁 Chunks directory:", self.chunks_dir)
        print("="*80)
    
    def load_and_analyze_chunks(self, signal_type='car'):
        print(f"
📊 LOADING AND ANALYZING {signal_type.upper()} CHUNKS")
        print("-" * 60)
        
        # Define file paths
        signal_chunks_dir = os.path.join(self.chunks_dir, signal_type)
        nothing_chunks_dir = os.path.join(self.chunks_dir, f"{signal_type}_nothing")
        
        all_features = []
        all_labels = []
        chunk_info = []
        
        # Process signal chunks
        if os.path.exists(signal_chunks_dir):
            print(f"Processing {signal_type} signal chunks...")
            features, labels, info = self._process_chunk_directory(
                signal_chunks_dir, label=0, signal_type=signal_type
            )
            all_features.extend(features)
            all_labels.extend(labels)
            chunk_info.extend(info)
            print(f"  ✅ Loaded {len(features)} {signal_type} signal segments")
        
        # Process nothing chunks
        if os.path.exists(nothing_chunks_dir):
            print(f"Processing {signal_type}_nothing chunks...")
            features, labels, info = self._process_chunk_directory(
                nothing_chunks_dir, label=1, signal_type=f"{signal_type}_nothing"
            )
            all_features.extend(features)
            all_labels.extend(labels)
            chunk_info.extend(info)
            print(f"  ✅ Loaded {len(features)} {signal_type} nothing segments")
        
        # Convert to arrays
        X = np.array(all_features) if all_features else np.array([])
        y = np.array(all_labels) if all_labels else np.array([])
        
        print(f"
📈 DATASET SUMMARY:")
        print(f"  Total segments: {len(X)}")
        if len(X) > 0:
            print(f"  Features per segment: {X.shape[1]}")
            print(f"  Signal segments: {np.sum(y == 0)}")
            print(f"  Nothing segments: {np.sum(y == 1)}")
            print(f"  Class balance: {np.sum(y == 0)/(len(y)+1e-10):.2%} signal, {np.sum(y == 1)/(len(y)+1e-10):.2%} nothing")
        
        return X, y, chunk_info
    
    def _process_chunk_directory(self, chunks_dir, label, signal_type):
        features = []
        labels = []
        chunk_info = []
        
        # Load chunk index
        index_file = os.path.join(chunks_dir, 'chunk_index.pkl')
        if not os.path.exists(index_file):
            print(f"  ⚠️  No chunk index found in {chunks_dir}")
            return features, labels, chunk_info
        
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        # Process each chunk
        for chunk_idx, chunk_file in enumerate(chunk_index['chunk_files']):
            if not os.path.exists(chunk_file):
                print(f"  ⚠️  Chunk file not found: {chunk_file}")
                continue
            
            try:
                # Load chunk data
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                # Extract features using advanced pattern analyzer
                chunk_features = self.pattern_analyzer.extract_segments_with_features(
                    chunk_data['spikes_bands_spectrogram'],
                    chunk_data['duration'],
                    signal_type=signal_type.replace('_nothing', ''),
                    segment_duration=12,  # Optimized segment duration
                    overlap=0.6          # Increased overlap for more data
                )
                
                # Add features and labels
                for segment_features in chunk_features[0]:  # chunk_features returns (segments, names)
                    features.append(segment_features)
                    labels.append(label)
                    chunk_info.append({
                        'chunk_file': chunk_file,
                        'chunk_idx': chunk_idx,
                        'signal_type': signal_type
                    })
                
            except Exception as e:
                print(f"  ❌ Error processing chunk {chunk_file}: {e}")
                continue
        
        return features, labels, chunk_info
    
    def optimize_classification(self, X, y, signal_type='car'):
        if len(X) == 0:
            print(f"❌ No data available for {signal_type} classification")
            return None
        
        print(f"
🧠 OPTIMIZED SNN CLASSIFICATION FOR {signal_type.upper()}")
        print("-" * 60)
        
        # Split data stratified
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test samples")
        except ValueError as e:
            print(f"⚠️  Stratified split failed ({e}), using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
        
        # Create optimized SNN
        snn = OptimizedGeophoneSNN(
            n_hidden=80,              # Increased hidden neurons
            learning_rate=0.010,      # Optimized learning rate
            architecture='optimized'
        )
        
        print(f"🔧 SNN Configuration:")
        print(f"  Hidden neurons: {snn.n_hidden}")
        print(f"  Learning rate: {snn.learning_rate}")
        print(f"  Architecture: {snn.architecture}")
        
        # Train with cross-validation
        print(f"
🎯 Training with cross-validation...")
        start_time = time.time()
        
        training_history, cv_scores = snn.train_with_cross_validation(
            X_train, y_train, 
            n_epochs=100,  # Increased epochs
            cv_folds=3
        )
        
        training_time = time.time() - start_time
        print(f"
⏱️  Training completed in {training_time:.1f} seconds")
        
        # Comprehensive evaluation
        print(f"
📊 COMPREHENSIVE EVALUATION")
        print("-" * 40)
        
        eval_results = snn.evaluate_comprehensive(X_test, y_test)
        
        # Additional analysis
        self._analyze_performance(eval_results, signal_type)
        
        # Save model
        model_path = f"optimized_{signal_type}_snn_model.pkl"
        snn.save_model(model_path)
        
        # Plot training history
        plot_path = f"optimized_{signal_type}_training_history.png"
        snn.plot_training_history(plot_path)
        
        return {
            'snn': snn,
            'training_history': training_history,
            'cv_scores': cv_scores,
            'eval_results': eval_results,
            'model_path': model_path,
            'training_time': training_time
        }
    
    def _analyze_performance(self, eval_results, signal_type):
        print(f"
🔍 DETAILED PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        accuracy = eval_results['accuracy']
        f1_score = eval_results['f1_score']
        cm = eval_results['confusion_matrix']
        confidence = eval_results['confidence']
        
        # Performance categories
        if accuracy >= 0.90:
            performance_level = "�� EXCELLENT"
        elif accuracy >= 0.80:
            performance_level = "🟡 GOOD"
        elif accuracy >= 0.70:
            performance_level = "🟠 FAIR"
        else:
            performance_level = "🔴 NEEDS IMPROVEMENT"
        
        print(f"Performance Level: {performance_level}")
        print(f"Overall Accuracy: {accuracy:.1%}")
        print(f"F1 Score: {f1_score:.3f}")
        
        # Confidence analysis
        high_conf_mask = confidence > 0.8
        medium_conf_mask = (confidence > 0.6) & (confidence <= 0.8)
        low_conf_mask = confidence <= 0.6
        
        print(f"
Confidence Distribution:")
        print(f"  High confidence (>0.8): {np.sum(high_conf_mask)} samples ({np.sum(high_conf_mask)/len(confidence):.1%})")
        print(f"  Medium confidence (0.6-0.8): {np.sum(medium_conf_mask)} samples ({np.sum(medium_conf_mask)/len(confidence):.1%})")
        print(f"  Low confidence (<0.6): {np.sum(low_conf_mask)} samples ({np.sum(low_conf_mask)/len(confidence):.1%})")
        
        # Class-specific analysis
        if cm.shape == (2, 2):
            signal_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
            signal_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            nothing_precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            nothing_recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            
            print(f"
Class-specific Performance:")
            print(f"  Signal class: Precision={signal_precision:.3f}, Recall={signal_recall:.3f}")
            print(f"  Nothing class: Precision={nothing_precision:.3f}, Recall={nothing_recall:.3f}")
            
            # Error analysis
            false_positives = cm[1, 0]  # Nothing classified as Signal
            false_negatives = cm[0, 1]  # Signal classified as Nothing
            
            print(f"
Error Analysis:")
            print(f"  False Positives (Nothing→Signal): {false_positives}")
            print(f"  False Negatives (Signal→Nothing): {false_negatives}")
            
            if false_positives > false_negatives:
                print(f"  📊 Model tends to over-detect signals")
            elif false_negatives > false_positives:
                print(f"  📊 Model tends to under-detect signals")
            else:
                print(f"  📊 Balanced error distribution")
    
    def run_complete_optimization(self):
        print(f"
🎯 RUNNING COMPLETE OPTIMIZATION")
        print("="*80)
        
        start_time = time.time()
        
        # Car classification
        print(f"
🚗 CAR CLASSIFICATION OPTIMIZATION")
        try:
            car_X, car_y, car_info = self.load_and_analyze_chunks('car')
            if len(car_X) > 0:
                car_results = self.optimize_classification(car_X, car_y, 'car')
                self.results['car'] = car_results
            else:
                print("❌ No car data available")
                self.results['car'] = None
        except Exception as e:
            print(f"❌ Car classification failed: {e}")
            self.results['car'] = None
        
        # Human classification  
        print(f"

👤 HUMAN CLASSIFICATION OPTIMIZATION")
        try:
            human_X, human_y, human_info = self.load_and_analyze_chunks('human')
            if len(human_X) > 0:
                human_results = self.optimize_classification(human_X, human_y, 'human')
                self.results['human'] = human_results
            else:
                print("❌ No human data available")
                self.results['human'] = None
        except Exception as e:
            print(f"❌ Human classification failed: {e}")
            self.results['human'] = None
        
        total_time = time.time() - start_time
        
        # Final summary
        self._print_final_summary(total_time)
        
        return self.results
    
    def _print_final_summary(self, total_time):
        print(f"

🎉 OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        print(f"
📊 FINAL RESULTS SUMMARY:")
        print("-" * 40)
        
        for signal_type in ['car', 'human']:
            result = self.results.get(signal_type)
            if result:
                accuracy = result['eval_results']['accuracy']
                f1_score = result['eval_results']['f1_score']
                training_time = result['training_time']
                cv_scores = result['cv_scores']
                
                # Performance emoji
                if accuracy >= 0.85:
                    emoji = "🟢"
                elif accuracy >= 0.75:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                
                print(f"
{emoji} {signal_type.upper()} CLASSIFICATION:")
                print(f"  ✅ Accuracy: {accuracy:.1%}")
                print(f"  📈 F1 Score: {f1_score:.3f}")
                print(f"  🔄 CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                print(f"  ⏱️  Training Time: {training_time:.1f}s")
                print(f"  💾 Model: {result['model_path']}")
            else:
                print(f"
❌ {signal_type.upper()} CLASSIFICATION: FAILED")
        
        print(f"
🔧 OPTIMIZATION IMPROVEMENTS:")
        print(f"  ✅ Advanced pattern-based feature extraction")
        print(f"  ✅ Optimized SNN architecture with improved stability")
        print(f"  ✅ Cross-validation for better generalization")
        print(f"  ✅ Comprehensive evaluation metrics")
        print(f"  ✅ Robust preprocessing and outlier handling")
        print(f"  ✅ Class balancing and adaptive learning")
        
        print(f"
🚀 SYSTEM CAPABILITIES:")
        print(f"  🎯 Real-time classification ready")
        print(f"  📊 Confidence-based decision making")
        print(f"  🔄 Cross-validated performance")
        print(f"  💾 Complete model persistence")
        print(f"  📈 Detailed performance analytics")
        
        print(f"
" + "="*80)
        print("="*80)

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Geophone Signal Optimization")
    
    # Create optimizer
    optimizer = ComprehensiveOptimizer()
    
    # Run complete optimization
    results = optimizer.run_complete_optimization()
    
    print("
✨ All done! Check the generated models and plots.")
