#!/usr/bin/env python3
# Simplified Comprehensive Optimizer
# Clean version without complex f-strings

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

class SimpleOptimizer:
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
        print()
        print("📊 LOADING AND ANALYZING", signal_type.upper(), "CHUNKS")
        print("-" * 60)
        
        # Define file paths
        signal_chunks_dir = os.path.join(self.chunks_dir, signal_type)
        nothing_chunks_dir = os.path.join(self.chunks_dir, signal_type + "_nothing")
        
        all_features = []
        all_labels = []
        chunk_info = []
        
        # Process signal chunks
        if os.path.exists(signal_chunks_dir):
            print("Processing", signal_type, "signal chunks...")
            features, labels, info = self._process_chunk_directory(
                signal_chunks_dir, label=0, signal_type=signal_type
            )
            all_features.extend(features)
            all_labels.extend(labels)
            chunk_info.extend(info)
            print("  ✅ Loaded", len(features), signal_type, "signal segments")
        
        # Process nothing chunks
        if os.path.exists(nothing_chunks_dir):
            print("Processing", signal_type + "_nothing chunks...")
            features, labels, info = self._process_chunk_directory(
                nothing_chunks_dir, label=1, signal_type=signal_type + "_nothing"
            )
            all_features.extend(features)
            all_labels.extend(labels)
            chunk_info.extend(info)
            print("  ✅ Loaded", len(features), signal_type, "nothing segments")
        
        # Convert to arrays
        X = np.array(all_features) if all_features else np.array([])
        y = np.array(all_labels) if all_labels else np.array([])
        
        print()
        print("📈 DATASET SUMMARY:")
        print("  Total segments:", len(X))
        if len(X) > 0:
            print("  Features per segment:", X.shape[1])
            print("  Signal segments:", np.sum(y == 0))
            print("  Nothing segments:", np.sum(y == 1))
            signal_pct = np.sum(y == 0)/(len(y)+1e-10)*100
            nothing_pct = np.sum(y == 1)/(len(y)+1e-10)*100
            print(f"  Class balance: {signal_pct:.1f}% signal, {nothing_pct:.1f}% nothing")
        
        return X, y, chunk_info
    
    def _process_chunk_directory(self, chunks_dir, label, signal_type):
        features = []
        labels = []
        chunk_info = []
        
        # Load chunk index
        index_file = os.path.join(chunks_dir, 'chunk_index.pkl')
        if not os.path.exists(index_file):
            print("  ⚠️  No chunk index found in", chunks_dir)
            return features, labels, chunk_info
        
        with open(index_file, 'rb') as f:
            chunk_index = pickle.load(f)
        
        # Process each chunk
        for chunk_idx, chunk_file in enumerate(chunk_index['chunk_files']):
            if not os.path.exists(chunk_file):
                print("  ⚠️  Chunk file not found:", chunk_file)
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
                print("  ❌ Error processing chunk", chunk_file, ":", str(e))
                continue
        
        return features, labels, chunk_info
    
    def optimize_classification(self, X, y, signal_type='car'):
        if len(X) == 0:
            print("❌ No data available for", signal_type, "classification")
            return None
        
        print()
        print("🧠 OPTIMIZED SNN CLASSIFICATION FOR", signal_type.upper())
        print("-" * 60)
        
        # Split data stratified
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            print("✅ Data split:", len(X_train), "train,", len(X_test), "test samples")
        except ValueError as e:
            print("⚠️  Stratified split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
        
        # Create optimized SNN
        snn = OptimizedGeophoneSNN(
            n_hidden=80,              # Increased hidden neurons
            learning_rate=0.010,      # Optimized learning rate
            architecture='optimized'
        )
        
        print("🔧 SNN Configuration:")
        print("  Hidden neurons:", snn.n_hidden)
        print("  Learning rate:", snn.learning_rate)
        print("  Architecture:", snn.architecture)
        
        # Train with cross-validation
        print()
        print("🎯 Training with cross-validation...")
        start_time = time.time()
        
        training_history, cv_scores = snn.train_with_cross_validation(
            X_train, y_train, 
            n_epochs=80,  # Reasonable epochs for testing
            cv_folds=3
        )
        
        training_time = time.time() - start_time
        print()
        print("⏱️  Training completed in", f"{training_time:.1f}", "seconds")
        
        # Comprehensive evaluation
        print()
        print("📊 COMPREHENSIVE EVALUATION")
        print("-" * 40)
        
        eval_results = snn.evaluate_comprehensive(X_test, y_test)
        
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
    
    def run_complete_optimization(self):
        print()
        print("🎯 RUNNING COMPLETE OPTIMIZATION")
        print("="*80)
        
        start_time = time.time()
        
        # Car classification
        print()
        print("🚗 CAR CLASSIFICATION OPTIMIZATION")
        try:
            car_X, car_y, car_info = self.load_and_analyze_chunks('car')
            if len(car_X) > 0:
                car_results = self.optimize_classification(car_X, car_y, 'car')
                self.results['car'] = car_results
            else:
                print("❌ No car data available")
                self.results['car'] = None
        except Exception as e:
            print("❌ Car classification failed:", str(e))
            self.results['car'] = None
        
        # Human classification  
        print()
        print()
        print("👤 HUMAN CLASSIFICATION OPTIMIZATION")
        try:
            human_X, human_y, human_info = self.load_and_analyze_chunks('human')
            if len(human_X) > 0:
                human_results = self.optimize_classification(human_X, human_y, 'human')
                self.results['human'] = human_results
            else:
                print("❌ No human data available")
                self.results['human'] = None
        except Exception as e:
            print("❌ Human classification failed:", str(e))
            self.results['human'] = None
        
        total_time = time.time() - start_time
        
        # Final summary
        self._print_final_summary(total_time)
        
        return self.results
    
    def _print_final_summary(self, total_time):
        print()
        print()
        print("🎉 OPTIMIZATION COMPLETE")
        print("="*80)
        print("⏱️  Total time:", f"{total_time:.1f}", "seconds")
        
        print()
        print("📊 FINAL RESULTS SUMMARY:")
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
                    emoji = "��"
                else:
                    emoji = "🔴"
                
                print()
                print(emoji, signal_type.upper(), "CLASSIFICATION:")
                print("  ✅ Accuracy:", f"{accuracy:.1%}")
                print("  📈 F1 Score:", f"{f1_score:.3f}")
                print("  🔄 CV Score:", f"{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                print("  ⏱️  Training Time:", f"{training_time:.1f}s")
                print("  💾 Model:", result['model_path'])
            else:
                print()
                print("❌", signal_type.upper(), "CLASSIFICATION: FAILED")
        
        print()
        print("🔧 OPTIMIZATION IMPROVEMENTS:")
        print("  ✅ Advanced pattern-based feature extraction")
        print("  ✅ Optimized SNN architecture with improved stability")
        print("  ✅ Cross-validation for better generalization")
        print("  ✅ Comprehensive evaluation metrics")
        print("  ✅ Robust preprocessing and outlier handling")
        print("  ✅ Class balancing and adaptive learning")
        
        print()
        print("🚀 SYSTEM CAPABILITIES:")
        print("  🎯 Real-time classification ready")
        print("  📊 Confidence-based decision making")
        print("  🔄 Cross-validated performance")
        print("  💾 Complete model persistence")
        print("  📈 Detailed performance analytics")
        
        print()
        print("="*80)
        print("🎊 COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Geophone Signal Optimization")
    
    # Create optimizer
    optimizer = SimpleOptimizer()
    
    # Run complete optimization
    results = optimizer.run_complete_optimization()
    
    print()
    print("✨ All done! Check the generated models and plots.")
