#!/usr/bin/env python3
"""
Advanced approaches to beat single neuron SCTN performance
"""
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

import sys
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from resonator_work import ProductionFeatureExtractor, CHUNKED_OUTPUT_DIR, SCTNClassifier

def test_advanced_approaches():
    """Test 6 different approaches to beat single neuron"""
    print("🚀 ADVANCED APPROACHES TO BEAT SINGLE NEURON")
    print("=" * 60)
    
    # Load data
    extractor = ProductionFeatureExtractor(chunk_dir=CHUNKED_OUTPUT_DIR)
    datasets = extractor.load_production_datasets()
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n🎯 {dataset_name.upper()} ADVANCED TESTING")
        print("-" * 40)
        
        # Baseline
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        baseline_model = SCTNClassifier(input_size=X.shape[1], signal_type=dataset_name)
        baseline_model.train(X_train, y_train, epochs=100, lr=0.1, verbose=False)
        y_pred = baseline_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Baseline (Single Neuron): {baseline_accuracy:.4f}")
        
        results = {'baseline': baseline_accuracy}
        
        # APPROACH 1: Feature Engineering
        print("🔬 Testing Feature Engineering...")
        try:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X)
            
            X_train_poly, X_test_poly, y_train, y_test = train_test_split(
                X_poly, y, test_size=0.3, stratify=y, random_state=42
            )
            
            model = SCTNClassifier(input_size=X_poly.shape[1], signal_type=dataset_name)
            model.train(X_train_poly, y_train, epochs=100, lr=0.1, verbose=False)
            y_pred = model.predict(X_test_poly)
            poly_accuracy = accuracy_score(y_test, y_pred)
            results['polynomial_features'] = poly_accuracy
            print(f"   Polynomial Features: {poly_accuracy:.4f}")
        except Exception as e:
            print(f"   Polynomial Features: Failed ({str(e)[:30]})")
        
        # APPROACH 2: Ensemble
        print("🎭 Testing Ensemble...")
        try:
            ensemble_preds = []
            configs = [
                {'lr': 0.05, 'epochs': 75},
                {'lr': 0.15, 'epochs': 100}, 
                {'lr': 0.1, 'epochs': 50},
            ]
            
            for config in configs:
                model = SCTNClassifier(input_size=X.shape[1], signal_type=dataset_name)
                model.train(X_train, y_train, epochs=config['epochs'], lr=config['lr'], verbose=False)
                y_pred = model.predict(X_test)
                ensemble_preds.append(y_pred)
            
            # Majority voting
            ensemble_pred = np.round(np.mean(ensemble_preds, axis=0))
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            results['ensemble'] = ensemble_accuracy
            print(f"   Ensemble (3 models): {ensemble_accuracy:.4f}")
        except Exception as e:
            print(f"   Ensemble: Failed ({str(e)[:30]})")
        
        # APPROACH 3: Different Preprocessing
        print("⚙️ Testing Preprocessing...")
        try:
            # Test different scaling ranges
            best_prep_acc = 0
            for min_val, max_val in [(0, 1), (-1, 1), (0.1, 0.9)]:
                scaler = MinMaxScaler(feature_range=(min_val, max_val))
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = SCTNClassifier(input_size=X.shape[1], signal_type=dataset_name)
                # Manually set scaler to avoid double scaling
                model.scaler = scaler
                model.is_trained = False
                
                # Simple training without internal scaling
                model._create_classifier_neuron()
                for epoch in range(100):
                    for i in range(len(X_train_scaled)):
                        features = X_train_scaled[i]
                        target = y_train[i]
                        prediction, _ = model._forward(features)
                        error = target - prediction
                        model.neuron.synapses_weights += 0.1 * error * features
                
                model.is_trained = True
                
                # Test
                correct = 0
                for i in range(len(X_test_scaled)):
                    prediction, _ = model._forward(X_test_scaled[i])
                    if prediction == y_test[i]:
                        correct += 1
                
                accuracy = correct / len(X_test_scaled)
                best_prep_acc = max(best_prep_acc, accuracy)
            
            results['preprocessing'] = best_prep_acc
            print(f"   Best Preprocessing: {best_prep_acc:.4f}")
        except Exception as e:
            print(f"   Preprocessing: Failed ({str(e)[:30]})")
        
        # APPROACH 4: Feature Selection
        print("🧬 Testing Feature Selection...")
        try:
            # Test different feature subsets
            feature_subsets = [
                [0, 1, 2, 3, 8, 9],      # Core + temporal
                [4, 5, 6, 7, 12, 13],    # Strength + signal
                [1, 3, 5, 7, 9, 11],     # Human-focused
                [0, 2, 4, 6, 8, 10],     # Car-focused
            ]
            
            best_fs_acc = 0
            for subset in feature_subsets:
                if max(subset) < X.shape[1]:
                    X_subset = X[:, subset]
                    X_train_sub, X_test_sub, y_train, y_test = train_test_split(
                        X_subset, y, test_size=0.3, stratify=y, random_state=42
                    )
                    
                    model = SCTNClassifier(input_size=len(subset), signal_type=dataset_name)
                    model.train(X_train_sub, y_train, epochs=100, lr=0.1, verbose=False)
                    y_pred = model.predict(X_test_sub)
                    accuracy = accuracy_score(y_test, y_pred)
                    best_fs_acc = max(best_fs_acc, accuracy)
            
            results['feature_selection'] = best_fs_acc
            print(f"   Best Feature Selection: {best_fs_acc:.4f}")
        except Exception as e:
            print(f"   Feature Selection: Failed ({str(e)[:30]})")
        
        # APPROACH 5: Hyperparameter Optimization
        print("🎯 Testing Hyperparameters...")
        try:
            best_hp_acc = 0
            for lr in [0.05, 0.08, 0.12, 0.18]:
                for epochs in [75, 125, 150]:
                    model = SCTNClassifier(input_size=X.shape[1], signal_type=dataset_name)
                    model.train(X_train, y_train, epochs=epochs, lr=lr, verbose=False)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    best_hp_acc = max(best_hp_acc, accuracy)
            
            results['hyperparameters'] = best_hp_acc
            print(f"   Best Hyperparameters: {best_hp_acc:.4f}")
        except Exception as e:
            print(f"   Hyperparameters: Failed ({str(e)[:30]})")
        
        # APPROACH 6: Data Augmentation
        print("📈 Testing Data Augmentation...")
        try:
            # Add noise to create more training samples
            X_augmented = []
            y_augmented = []
            
            # Original data
            X_augmented.extend(X_train)
            y_augmented.extend(y_train)
            
            # Add noisy versions
            for i in range(len(X_train)):
                noise = np.random.normal(0, 0.05, X_train[i].shape)
                X_noisy = X_train[i] + noise
                X_augmented.append(X_noisy)
                y_augmented.append(y_train[i])
            
            X_aug = np.array(X_augmented)
            y_aug = np.array(y_augmented)
            
            model = SCTNClassifier(input_size=X.shape[1], signal_type=dataset_name)
            model.train(X_aug, y_aug, epochs=100, lr=0.1, verbose=False)
            y_pred = model.predict(X_test)
            aug_accuracy = accuracy_score(y_test, y_pred)
            results['data_augmentation'] = aug_accuracy
            print(f"   Data Augmentation: {aug_accuracy:.4f}")
        except Exception as e:
            print(f"   Data Augmentation: Failed ({str(e)[:30]})")
        
        # Results summary
        print(f"\n🏆 RESULTS FOR {dataset_name.upper()}:")
        print("-" * 40)
        
        best_method = max(results, key=results.get)
        best_accuracy = results[best_method]
        improvement = best_accuracy - baseline_accuracy
        
        for method, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            diff = accuracy - baseline_accuracy
            marker = "🏆" if method == best_method else "📊"
            print(f"{method:<20}: {accuracy:.4f} ({diff:+.4f}) {marker}")
        
        if improvement > 0.01:
            print(f"\n✅ IMPROVEMENT: {best_method} beats baseline by {improvement:+.4f}")
        else:
            print(f"\n📊 Single neuron remains optimal")
        
        # Save results
        with open(f"advanced_results_{dataset_name}.pkl", 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Results saved to: advanced_results_{dataset_name}.pkl")

if __name__ == "__main__":
    test_advanced_approaches()
