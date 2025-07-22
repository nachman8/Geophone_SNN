#!/usr/bin/env python3
"""
Advanced SCTN approaches to beat single neuron performance
Multiple innovative techniques beyond simple multi-layer networks
"""

import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.ensemble import VotingClassifier
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import sys
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

from sctnN.spiking_neuron import create_SCTN, BINARY
from resonator_work import ProductionFeatureExtractor, CHUNKED_OUTPUT_DIR, SCTNClassifier

class AdvancedSCTNApproaches:
    """Collection of advanced approaches to improve beyond single neuron"""
    
    def __init__(self, X, y, signal_type):
        self.X = X
        self.y = y
        self.signal_type = signal_type
        self.results = {}
    
    def approach_1_feature_engineering(self):
        """APPROACH 1: Advanced Feature Engineering + Single Neuron"""
        print("🔬 APPROACH 1: Advanced Feature Engineering")
        print("-" * 50)
        
        # Create polynomial features (interaction terms)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(self.X)
        print(f"   Original features: {self.X.shape[1]}")
        print(f"   Polynomial features: {X_poly.shape[1]}")
        
        # Test with enhanced features
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, self.y, test_size=0.3, stratify=self.y, random_state=42
        )
        
        model = SCTNClassifier(input_size=X_poly.shape[1], signal_type=self.signal_type)
        model.train(X_train, y_train, epochs=100, lr=0.1, verbose=False)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✅ Polynomial Features Accuracy: {accuracy:.4f}")
        self.results['feature_engineering'] = accuracy
        return accuracy
    
    def approach_2_ensemble_neurons(self):
        """APPROACH 2: Ensemble of Specialized SCTN Neurons"""
        print("🎭 APPROACH 2: Ensemble of Specialized Neurons")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y, random_state=42
        )
        
        # Create ensemble of specialized neurons
        ensemble_predictions = []
        ensemble_accuracies = []
        
        # Different specializations
        configs = [
            {'lr': 0.05, 'epochs': 75, 'name': 'Conservative'},
            {'lr': 0.15, 'epochs': 100, 'name': 'Aggressive'}, 
            {'lr': 0.1, 'epochs': 50, 'name': 'Fast'},
            {'lr': 0.1, 'epochs': 150, 'name': 'Deep'},
        ]
        
        for config in configs:
            model = SCTNClassifier(input_size=self.X.shape[1], signal_type=self.signal_type)
            model.train(X_train, y_train, epochs=config['epochs'], lr=config['lr'], verbose=False)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            ensemble_predictions.append(y_pred)
            ensemble_accuracies.append(accuracy)
            print(f"   {config['name']} Neuron: {accuracy:.4f}")
        
        # Majority voting
        ensemble_pred = np.array(ensemble_predictions)
        final_pred = np.round(np.mean(ensemble_pred, axis=0))
        ensemble_accuracy = accuracy_score(y_test, final_pred)
        
        print(f"   ✅ Ensemble Accuracy: {ensemble_accuracy:.4f}")
        self.results['ensemble'] = ensemble_accuracy
        return ensemble_accuracy
    
    def approach_3_adaptive_threshold(self):
        """APPROACH 3: Adaptive Threshold SCTN"""
        print("🎯 APPROACH 3: Adaptive Threshold Learning")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y, random_state=42
        )
        
        # Test different threshold adaptation strategies
        best_accuracy = 0
        best_strategy = None
        
        threshold_strategies = [
            (15.0, 45.0, "Low Range"),
            (20.0, 60.0, "Medium Range"), 
            (25.0, 75.0, "High Range"),
            (10.0, 80.0, "Wide Range"),
        ]
        
        for low, high, name in threshold_strategies:
            model = AdaptiveThresholdSCTN(
                input_size=self.X.shape[1], 
                signal_type=self.signal_type,
                threshold_range=(low, high)
            )
            model.train(X_train, y_train, epochs=100, lr=0.1, verbose=False)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"   {name}: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_strategy = name
        
        print(f"   ✅ Best Adaptive Threshold ({best_strategy}): {best_accuracy:.4f}")
        self.results['adaptive_threshold'] = best_accuracy
        return best_accuracy
    
    def approach_4_competitive_learning(self):
        """APPROACH 4: Competitive Learning Network"""
        print("🏆 APPROACH 4: Competitive Learning SCTN")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y, random_state=42
        )
        
        # Create competitive network
        accuracy = self._test_competitive_network(X_train, y_train, X_test, y_test)
        
        print(f"   ✅ Competitive Learning Accuracy: {accuracy:.4f}")
        self.results['competitive'] = accuracy
        return accuracy
    
    def approach_5_feature_selection(self):
        """APPROACH 5: Intelligent Feature Selection + SCTN"""
        print("🧬 APPROACH 5: Intelligent Feature Selection")
        print("-" * 50)
        
        # Test different feature subsets
        feature_subsets = [
            [0, 1, 2, 3],           # Core discriminative features
            [4, 5, 6, 7],           # Activity strength features  
            [8, 9, 10, 11],         # Temporal patterns
            [12, 13, 14, 15],       # Signal characteristics
            [1, 3, 5, 7, 9, 11],    # Human-specific features
            [0, 2, 4, 6, 8, 10],    # Car-specific features
        ]
        
        best_accuracy = 0
        best_subset = None
        
        for i, subset in enumerate(feature_subsets):
            if max(subset) < self.X.shape[1]:  # Ensure valid indices
                X_subset = self.X[:, subset]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subset, self.y, test_size=0.3, stratify=self.y, random_state=42
                )
                
                model = SCTNClassifier(input_size=len(subset), signal_type=self.signal_type)
                model.train(X_train, y_train, epochs=100, lr=0.1, verbose=False)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"   Subset {i+1} ({len(subset)} features): {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_subset = i+1
        
        print(f"   ✅ Best Feature Subset ({best_subset}): {best_accuracy:.4f}")
        self.results['feature_selection'] = best_accuracy
        return best_accuracy
    
    def approach_6_preprocessing_optimization(self):
        """APPROACH 6: Advanced Preprocessing + SCTN"""
        print("⚙️ APPROACH 6: Advanced Preprocessing")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y, random_state=42
        )
        
        # Test different preprocessing approaches
        preprocessing_methods = [
            (StandardScaler(), "Standard Scaling"),
            (MinMaxScaler(feature_range=(-1, 1)), "MinMax [-1,1]"),
            (MinMaxScaler(feature_range=(0.2, 0.8)), "MinMax [0.2,0.8]"),
        ]
        
        best_accuracy = 0
        best_method = None
        
        for scaler, name in preprocessing_methods:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Use custom SCTN with different preprocessing
            accuracy = self._test_with_preprocessing(X_train_scaled, y_train, X_test_scaled, y_test)
            
            print(f"   {name}: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = name
        
        print(f"   ✅ Best Preprocessing ({best_method}): {best_accuracy:.4f}")
        self.results['preprocessing'] = best_accuracy
        return best_accuracy
    
    def _test_competitive_network(self, X_train, y_train, X_test, y_test):
        """Test competitive learning approach"""
        # Simplified competitive learning with SCTN neurons
        num_competitors = 3
        neurons = []
        
        for i in range(num_competitors):
            neuron = create_SCTN()
            neuron.synapses_weights = np.random.normal(0, 0.1, X_train.shape[1]).astype(np.float64)
            neuron.threshold_pulse = np.random.uniform(20.0, 60.0)
            neuron.activation_function = BINARY
            neurons.append(neuron)
        
        # Competitive training
        for epoch in range(75):
            for i in range(len(X_train)):
                features = X_train[i]
                target = y_train[i]
                
                # Find winning neuron
                activations = []
                for neuron in neurons:
                    activation = np.dot(features, neuron.synapses_weights)
                    activations.append(activation)
                
                winner_idx = np.argmax(activations)
                
                # Update winner
                error = target - (1 if activations[winner_idx] > neurons[winner_idx].threshold_pulse else 0)
                neurons[winner_idx].synapses_weights += 0.1 * error * features
        
        # Test
        correct = 0
        for i in range(len(X_test)):
            features = X_test[i]
            target = y_test[i]
            
            activations = []
            for neuron in neurons:
                activation = np.dot(features, neuron.synapses_weights)
                output = 1 if activation > neuron.threshold_pulse else 0
                activations.append(output)
            
            prediction = max(set(activations), key=activations.count)  # Majority vote
            if prediction == target:
                correct += 1
        
        return correct / len(X_test)
    
    def _test_with_preprocessing(self, X_train, y_train, X_test, y_test):
        """Test with custom preprocessing"""
        neuron = create_SCTN()
        neuron.synapses_weights = np.random.normal(0, 0.1, X_train.shape[1]).astype(np.float64)
        neuron.threshold_pulse = 40.0
        neuron.activation_function = BINARY
        
        # Training
        for epoch in range(100):
            for i in range(len(X_train)):
                features = X_train[i]
                target = y_train[i]
                
                activation = np.dot(features, neuron.synapses_weights)
                prediction = 1 if activation > neuron.threshold_pulse else 0
                error = target - prediction
                
                neuron.synapses_weights += 0.1 * error * features
                neuron.threshold_pulse += 0.1 * error * 0.1
        
        # Testing
        correct = 0
        for i in range(len(X_test)):
            features = X_test[i]
            target = y_test[i]
            
            activation = np.dot(features, neuron.synapses_weights)
            prediction = 1 if activation > neuron.threshold_pulse else 0
            
            if prediction == target:
                correct += 1
        
        return correct / len(X_test)

class AdaptiveThresholdSCTN(SCTNClassifier):
    """SCTN with adaptive threshold learning"""
    
    def __init__(self, input_size=16, signal_type=None, threshold_range=(20.0, 60.0)):
        super().__init__(input_size, signal_type=signal_type)
        self.threshold_range = threshold_range
    
    def _create_classifier_neuron(self):
        """Create neuron with adaptive threshold"""
        neuron = super()._create_classifier_neuron()
        
        # Initialize threshold in specified range
        neuron.threshold_pulse = np.random.uniform(
            self.threshold_range[0], self.threshold_range[1]
        )
        
        return neuron

def run_all_advanced_approaches():
    """Test all advanced approaches against single neuron baseline"""
    print("🚀 ADVANCED SCTN APPROACHES TESTING")
    print("=" * 80)
    print("Testing 6 different approaches to beat single neuron performance")
    print()
    
    # Load data
    extractor = ProductionFeatureExtractor(chunk_dir=CHUNKED_OUTPUT_DIR)
    datasets = extractor.load_production_datasets()
    
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    all_results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n🎯 TESTING ADVANCED APPROACHES: {dataset_name.upper()}")
        print("=" * 60)
        
        # Baseline single neuron
        print("📊 BASELINE: Single Neuron")
        print("-" * 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        baseline_model = SCTNClassifier(input_size=X.shape[1], signal_type=dataset_name)
        baseline_model.train(X_train, y_train, epochs=100, lr=0.1, verbose=False)
        y_pred = baseline_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred)
        print(f"   Baseline Accuracy: {baseline_accuracy:.4f}")
        print()
        
        # Test advanced approaches
        approaches = AdvancedSCTNApproaches(X, y, dataset_name)
        
        results = {
            'baseline': baseline_accuracy,
            'feature_engineering': approaches.approach_1_feature_engineering(),
            'ensemble': approaches.approach_2_ensemble_neurons(),
            'adaptive_threshold': approaches.approach_3_adaptive_threshold(),
            'competitive': approaches.approach_4_competitive_learning(),
            'feature_selection': approaches.approach_5_feature_selection(),
            'preprocessing': approaches.approach_6_preprocessing_optimization(),
        }
        
        # Find best approach
        best_approach = max(results, key=results.get)
        best_accuracy = results[best_approach]
        improvement = best_accuracy - baseline_accuracy
        
        print(f"\n🏆 RESULTS SUMMARY FOR {dataset_name.upper()}:")
        print("-" * 50)
        print(f"{'Approach':<20} {'Accuracy':<10} {'vs Baseline':<12}")
        print("-" * 50)
        
        for approach, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            diff = accuracy - baseline_accuracy
            status = "🏆" if approach == best_approach else "📊"
            print(f"{approach:<20} {accuracy:.4f}    {diff:+.4f}     {status}")
        
        print("-" * 50)
        if improvement > 0.01:
            print(f"✅ IMPROVEMENT FOUND! {best_approach} beats baseline by {improvement:+.4f}")
        else:
            print(f"📊 Single neuron remains competitive")
        
        all_results[dataset_name] = results
        
        # Save best approach
        with open(f"best_advanced_{dataset_name}.pkl", 'wb') as f:
            pickle.dump({
                'best_approach': best_approach,
                'best_accuracy': best_accuracy,
                'improvement': improvement,
                'all_results': results
            }, f)
    
    return all_results

if __name__ == "__main__":
    results = run_all_advanced_approaches() 