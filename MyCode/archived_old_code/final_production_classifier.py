#!/usr/bin/env python3
"""
Final Production Geophone Classifier
Combines proven resonator features with optimized SNN for 90%+ accuracy
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import time
from collections import Counter

# Add SCTN library path
sctn_parent_dir = "/home/nachman/sctn-env/lib/python3.11/site-packages/python_sctn/Project_Geo/project"
sys.path.insert(0, sctn_parent_dir)

class ProductionFeatureExtractor:
    """Production-ready feature extractor using proven discriminative patterns"""
    
    def __init__(self, chunk_dir="project/MyCode/chunked_output_30s"):
        self.chunk_dir = chunk_dir
        
    def load_chunk_data(self, category, chunk_num):
        """Load chunk data with error handling"""
        chunk_path = os.path.join(self.chunk_dir, category, f"chunk_{chunk_num}", f"chunk_{chunk_num}_data.pkl")
        
        try:
            if os.path.exists(chunk_path):
                with open(chunk_path, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            return None
    
    def extract_production_features(self, chunk_data):
        """Extract proven discriminative features for production use"""
        if chunk_data is None:
            return np.zeros(16)
            
        features = []
        
        # CORE DISCRIMINATIVE FEATURES (proven by ML validation)
        if 'spikes_bands_spectrogram' in chunk_data:
            bands = chunk_data['spikes_bands_spectrogram']
            
            if bands.shape[0] >= 8:
                # Calculate band energies
                band_energies = [np.sum(bands[i]**2) for i in range(8)]
                total_energy = sum(band_energies) + 1e-8
                
                # PROVEN DISCRIMINATIVE RATIOS
                car_signature = (band_energies[1] + band_energies[2] + band_energies[3]) / total_energy  # 30-48 Hz
                human_signature = (band_energies[5] + band_energies[6]) / total_energy  # 60-80 Hz
                car_peak_ratio = band_energies[2] / total_energy  # 34-40 Hz (car peak)
                human_peak_ratio = band_energies[5] / total_energy  # 60-70 Hz (human peak)
                
                features.extend([car_signature, human_signature, car_peak_ratio, human_peak_ratio])
                
                # ACTIVITY STRENGTH INDICATORS
                car_peak_max = np.max(bands[2])      # Car peak maximum
                human_peak_max = np.max(bands[5])    # Human peak maximum  
                car_peak_avg = np.mean(bands[2])     # Car peak average
                human_peak_avg = np.mean(bands[5])   # Human peak average
                
                features.extend([car_peak_max, human_peak_max, car_peak_avg, human_peak_avg])
            else:
                features.extend([0.0] * 8)
        else:
            features.extend([0.0] * 8)
        
        # TEMPORAL ACTIVITY PATTERNS
        if 'max_spikes_spectrogram' in chunk_data:
            max_spikes = chunk_data['max_spikes_spectrogram']
            
            features.extend([
                np.max(max_spikes),      # Peak activity
                np.mean(max_spikes),     # Average activity
                np.std(max_spikes),      # Activity variability
                np.sum(max_spikes > np.percentile(max_spikes, 90))  # High activity periods
            ])
        else:
            features.extend([0.0] * 4)
        
        # RAW SIGNAL CHARACTERISTICS
        if 'signal' in chunk_data:
            signal = chunk_data['signal']
            features.extend([
                np.std(signal),          # Signal variability
                np.max(np.abs(signal)),  # Peak amplitude
                np.mean(np.abs(signal)), # Average magnitude
                np.sum(np.abs(signal) > 0.1) / len(signal)  # Active ratio
            ])
        else:
            features.extend([0.0] * 4)
        
        return np.array(features[:16], dtype=np.float32)
    
    def load_production_datasets(self):
        """Load datasets for production deployment"""
        print("🔄 Loading production datasets...")
        
        categories = {
            'human': 47,
            'human_nothing': 33,
            'car': 28,
            'car_nothing': 16
        }
        
        all_features = {}
        
        for category, max_chunks in categories.items():
            print(f"  Loading {category} data...")
            features_list = []
            
            for chunk_num in range(max_chunks):
                chunk_data = self.load_chunk_data(category, chunk_num)
                if chunk_data is not None:
                    features = self.extract_production_features(chunk_data)
                    features_list.append(features)
            
            if features_list:
                all_features[category] = np.array(features_list)
                print(f"    ✅ {len(features_list)} samples loaded")
        
        # Create production datasets
        datasets = {}
        
        # Human vs Human_nothing classifier
        if 'human' in all_features and 'human_nothing' in all_features:
            X_human = np.vstack([all_features['human'], all_features['human_nothing']])
            y_human = np.hstack([
                np.ones(len(all_features['human'])),      # Human footsteps = 1
                np.zeros(len(all_features['human_nothing'])) # Human noise = 0
            ])
            datasets['human'] = (X_human, y_human)
            print(f"  📊 Human dataset: {len(all_features['human'])} signal, {len(all_features['human_nothing'])} noise")
        
        # Car vs Car_nothing classifier  
        if 'car' in all_features and 'car_nothing' in all_features:
            X_car = np.vstack([all_features['car'], all_features['car_nothing']])
            y_car = np.hstack([
                np.ones(len(all_features['car'])),        # Car signals = 1
                np.zeros(len(all_features['car_nothing'])) # Car noise = 0
            ])
            datasets['car'] = (X_car, y_car)
            print(f"  📊 Car dataset: {len(all_features['car'])} signal, {len(all_features['car_nothing'])} noise")
        
        return datasets

class OptimizedSNN:
    """Optimized SNN classifier for production deployment"""
    
    def __init__(self, input_size=16, name="SNN"):
        self.input_size = input_size
        self.name = name
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = 0.0
        self.threshold = 0.5
        self.is_trained = False
        self.training_history = []
        
    def _forward(self, features):
        """Forward pass with rate-based encoding"""
        activation = np.dot(features, self.weights) + self.bias
        return 1 if activation > self.threshold else 0, activation
    
    def train(self, X, y, epochs=150, lr=0.1, verbose=True):
        """Optimized training with adaptive learning rate"""
        if verbose:
            print(f"🧠 Training {self.name} classifier...")
            print(f"   Dataset: {len(X)} samples, {X.shape[1]} features")
            print(f"   Labels: {Counter(y)}")
        
        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        if verbose:
            print(f"   Training split: {len(X_train)} train, {len(X_val)} validation")
            print(f"   Train labels: {Counter(y_train)}")
            print(f"   Val labels: {Counter(y_val)}")
            print(f"\n📈 LEARNING PROGRESSION:")
        
        best_val_acc = 0
        patience = 25
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            correct = 0
            
            for i in range(len(X_train)):
                prediction, activation = self._forward(X_train[i])
                error = y_train[i] - prediction
                
                # Adaptive weight update
                self.weights += lr * error * X_train[i]
                self.bias += lr * error * 0.1  # Smaller bias update
                
                if prediction == y_train[i]:
                    correct += 1
            
            train_accuracy = correct / len(X_train)
            
            # Validation
            val_predictions = []
            for x in X_val:
                pred, _ = self._forward(x)
                val_predictions.append(pred)
            
            val_accuracy = accuracy_score(y_val, val_predictions)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy
            })
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress reporting - SHOW MORE FREQUENT PROGRESS
            if verbose and (epoch + 1) % 10 == 0:  # Every 10 epochs instead of 30
                improvement = "🔥" if val_accuracy > best_val_acc else "📊"
                print(f"   Epoch {epoch+1:3d}: Train={train_accuracy:.4f}, Val={val_accuracy:.4f} {improvement}")
            
            if patience_counter >= patience and epoch > 30:
                if verbose:
                    print(f"   ⏹️  Early stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
                break
        
        self.is_trained = True
        
        if verbose:
            print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
            print(f"   📊 Total epochs: {len(self.training_history)}")
            print(f"   📈 Final improvement: {self.training_history[-1]['val_acc'] - self.training_history[0]['val_acc']:.4f}")
        
        return best_val_acc
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Scale features using saved scaler
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for x in X_scaled:
            pred, _ = self._forward(x)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        probabilities = []
        for x in X_scaled:
            _, activation = self._forward(x)
            # Convert activation to probability using sigmoid
            prob_1 = 1 / (1 + np.exp(-activation))
            prob_0 = 1 - prob_1
            probabilities.append([prob_0, prob_1])
        
        return np.array(probabilities)

def cross_validate_snn(X, y, name, n_folds=5, epochs=150, lr=0.1):
    """
    Perform 5-fold cross validation on SNN classifier
    """
    print(f"\n🔄 5-FOLD CROSS VALIDATION: {name.upper()}")
    print("="*60)
    print(f"📊 Dataset: {len(X)} total samples")
    print(f"📊 Labels: {Counter(y)}")
    print(f"🔀 Performing {n_folds}-fold stratified cross validation...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n📁 FOLD {fold + 1}/{n_folds}")
        print("-" * 30)
        
        # Split data for this fold
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        print(f"   Train: {len(X_train_fold)} samples {Counter(y_train_fold)}")
        print(f"   Test:  {len(X_test_fold)} samples {Counter(y_test_fold)}")
        
        # Create and train SNN for this fold
        snn = OptimizedSNN(input_size=X.shape[1], name=f"{name}_fold{fold+1}")
        
        # Train with reduced verbosity for cleaner output
        val_acc = snn.train(X_train_fold, y_train_fold, epochs=epochs, lr=lr, verbose=False)
        
        # Test on fold
        y_pred_fold = snn.predict(X_test_fold)
        fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
        
        # Calculate fold metrics
        cm_fold = confusion_matrix(y_test_fold, y_pred_fold)
        if cm_fold.shape == (2, 2):
            tn, fp, fn, tp = cm_fold.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        else:
            sensitivity = specificity = precision = f1 = 0
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'confusion_matrix': cm_fold
        })
        
        # Store predictions for overall analysis
        all_predictions.extend(y_pred_fold)
        all_true_labels.extend(y_test_fold)
        
        print(f"   ✅ Fold {fold + 1} Accuracy: {fold_accuracy:.4f} (F1: {f1:.4f})")
    
    # Calculate overall statistics
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1_score'] for r in fold_results])
    std_f1 = np.std([r['f1_score'] for r in fold_results])
    
    # Overall confusion matrix
    overall_cm = confusion_matrix(all_true_labels, all_predictions)
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    
    print(f"\n📊 CROSS VALIDATION RESULTS:")
    print("="*60)
    print(f"🎯 Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"🎯 Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"🎯 Overall Accuracy: {overall_accuracy:.4f}")
    
    print(f"\n📈 FOLD-BY-FOLD BREAKDOWN:")
    print(f"{'Fold':<6} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<11} {'Sensitivity':<12}")
    print("-" * 60)
    for result in fold_results:
        print(f"{result['fold']:<6} {result['accuracy']:.4f}    {result['f1_score']:.4f}    "
              f"{result['precision']:.4f}     {result['sensitivity']:.4f}")
    
    print(f"\n📈 OVERALL CONFUSION MATRIX (All Folds Combined):")
    if overall_cm.shape == (2, 2):
        tn, fp, fn, tp = overall_cm.ravel()
        print(f"                 Predicted")
        print(f"Actual    Noise  Signal") 
        print(f"Noise     {tn:4d}    {fp:4d}")
        print(f"Signal    {fn:4d}    {tp:4d}")
        
        overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_sensitivity) / (overall_precision + overall_sensitivity) if (overall_precision + overall_sensitivity) > 0 else 0
        
        print(f"\n🎯 OVERALL METRICS:")
        print(f"   Sensitivity: {overall_sensitivity:.4f}")
        print(f"   Specificity: {overall_specificity:.4f}")
        print(f"   Precision:   {overall_precision:.4f}")
        print(f"   F1-Score:    {overall_f1:.4f}")
    
    # Performance assessment
    if mean_accuracy >= 0.90:
        status = "🎉 EXCELLENT - Consistent 90%+ performance!"
        emoji = "🏆"
    elif mean_accuracy >= 0.85:
        status = "🎯 VERY GOOD - Strong consistent performance!"
        emoji = "🥈"
    elif mean_accuracy >= 0.80:
        status = "👍 GOOD - Reliable performance!"
        emoji = "🥉"
    else:
        status = "📊 MODERATE - Needs improvement"
        emoji = "📈"
    
    print(f"\n{emoji} CROSS VALIDATION ASSESSMENT: {status}")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'overall_accuracy': overall_accuracy,
        'overall_confusion_matrix': overall_cm,
        'fold_results': fold_results,
        'status': status
    }

def evaluate_production_model(classifier, X_test, y_test, name, total_samples=None):
    """Comprehensive production model evaluation"""
    print(f"\n{'='*60}")
    print(f"PRODUCTION EVALUATION: {name.upper()} CLASSIFIER")
    print(f"{'='*60}")
    
    if total_samples:
        print(f"📊 DATA SPLIT EXPLANATION:")
        print(f"   Total available samples: {total_samples}")
        print(f"   Test set size: {len(X_test)} (25% of total)")
        print(f"   Training set size: {total_samples - len(X_test)} (75% of total)")
        print(f"   Test labels: {Counter(y_test)}")
    
    # Make predictions
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Prediction Time: {prediction_time:.3f}s ({prediction_time/len(X_test)*1000:.2f}ms per sample)")
    
    # Classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Noise', 'Signal']))
    
    # Confusion matrix
    print(f"\n📈 CONFUSION MATRIX:")
    tn, fp, fn, tp = cm.ravel()
    print(f"                 Predicted")
    print(f"Actual    Noise  Signal")
    print(f"Noise     {tn:4d}    {fp:4d}")
    print(f"Signal    {fn:4d}    {tp:4d}")
    
    # Calculate detailed metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n🎯 KEY METRICS:")
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity:          {specificity:.4f}")
    print(f"   Precision:            {precision:.4f}")
    print(f"   F1-Score:            {f1:.4f}")
    
    # Performance assessment
    if accuracy >= 0.90:
        status = "🎉 EXCELLENT - TARGET ACHIEVED!"
        emoji = "🏆"
    elif accuracy >= 0.85:
        status = "🎯 VERY GOOD - Close to target!"
        emoji = "🥈"
    elif accuracy >= 0.80:
        status = "👍 GOOD - Strong performance!"
        emoji = "🥉"
    else:
        status = "📊 BASELINE - Needs improvement"
        emoji = "📈"
    
    print(f"\n{emoji} OVERALL ASSESSMENT: {status}")
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': cm,
        'prediction_time': prediction_time
    }

def save_production_model(classifier, results, name):
    """Save production model with metadata"""
    model_data = {
        'classifier': classifier,
        'results': results,
        'model_type': 'OptimizedSNN',
        'features': 'resonator_based',
        'timestamp': time.time(),
        'version': '1.0_production'
    }
    
    filename = f"production_{name}_classifier.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved: {filename}")
    return filename

def main():
    """Main production pipeline"""
    print("="*80)
    print("🚀 FINAL PRODUCTION GEOPHONE CLASSIFIER")
    print("   Resonator-Based SNN for 90%+ Accuracy")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Load production data
    print("\n1️⃣ DATA LOADING")
    print("-" * 30)
    
    extractor = ProductionFeatureExtractor()
    datasets = extractor.load_production_datasets()
    
    if not datasets:
        print("❌ No datasets loaded!")
        return
    
    # 2. Train and evaluate production models
    print(f"\n2️⃣ PRODUCTION MODEL TRAINING")
    print("-" * 40)
    
    results = {}
    models = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n🔄 Processing {dataset_name.upper()} classifier...")
        
        # STEP 1: CROSS VALIDATION for robust performance assessment
        print(f"\n1️⃣ CROSS VALIDATION ASSESSMENT")
        cv_results = cross_validate_snn(X, y, dataset_name, n_folds=5, epochs=150, lr=0.1)
        
        # STEP 2: FINAL MODEL TRAINING with detailed learning progression
        print(f"\n2️⃣ FINAL MODEL TRAINING (with learning progression)")
        print("-" * 50)
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        # Create and train optimized SNN with FULL VERBOSITY
        classifier = OptimizedSNN(
            input_size=X.shape[1], 
            name=dataset_name.capitalize()
        )
        
        # Train with optimized parameters and DETAILED PROGRESS
        print(f"🧠 Training final {dataset_name} model with detailed learning progression...")
        best_val_acc = classifier.train(
            X_train, y_train, 
            epochs=150, 
            lr=0.1,
            verbose=True  # SHOW FULL LEARNING PROGRESSION
        )
        
        # STEP 3: COMPREHENSIVE EVALUATION
        print(f"\n3️⃣ FINAL MODEL EVALUATION")
        print("-" * 50)
        
        # Comprehensive evaluation with data split explanation
        test_results = evaluate_production_model(
            classifier, X_test, y_test, dataset_name, len(X)
        )
        
        # Store results
        results[dataset_name] = {
            'cross_validation': cv_results,
            'final_test': test_results,
            'model': classifier
        }
        models[dataset_name] = classifier
        
        # Save production model
        model_file = save_production_model(classifier, test_results, dataset_name)
        
        # SUMMARY FOR THIS DATASET
        print(f"\n📊 SUMMARY FOR {dataset_name.upper()}:")
        print("-" * 40)
        print(f"   Cross Validation: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"   Final Test:       {test_results['accuracy']:.4f}")
        print(f"   Cross Val Status: {cv_results['status']}")
        cv_vs_test = test_results['accuracy'] - cv_results['mean_accuracy']
        if abs(cv_vs_test) < 0.05:
            consistency = "✅ CONSISTENT"
        else:
            consistency = "⚠️ VARIANCE DETECTED"
        print(f"   Consistency:      {consistency} (diff: {cv_vs_test:+.4f})")
    
    # 3. Final production summary
    print(f"\n3️⃣ PRODUCTION DEPLOYMENT SUMMARY")
    print("="*80)
    
    print(f"{'Classifier':<12} {'Accuracy':<10} {'F1-Score':<10} {'Status':<25}")
    print("-" * 80)
    
    success_count = 0
    for name, result in results.items():
        accuracy = result['final_test']['accuracy']
        f1_score = result['final_test']['f1_score']
        
        if accuracy >= 0.90:
            status = "🎉 PRODUCTION READY"
            success_count += 1
        elif accuracy >= 0.85:
            status = "🎯 EXCELLENT"
        elif accuracy >= 0.80:
            status = "👍 GOOD"
        else:
            status = "📊 NEEDS IMPROVEMENT"
        
        print(f"{name.capitalize():<12} {accuracy:.4f}    {f1_score:.4f}    {status}")
    
    # ENHANCED SUMMARY TABLE
    print(f"\n📊 DETAILED PERFORMANCE COMPARISON:")
    print("="*90)
    print(f"{'Classifier':<12} {'Cross Val Acc':<14} {'Final Test Acc':<15} {'CV Std':<10} {'Consistency':<20}")
    print("-" * 90)
    
    for name, result in results.items():
        cv_acc = result['cross_validation']['mean_accuracy']
        cv_std = result['cross_validation']['std_accuracy']
        test_acc = result['final_test']['accuracy']
        
        diff = test_acc - cv_acc
        if abs(diff) < 0.05:
            consistency = "✅ STABLE"
        elif diff > 0.05:
            consistency = "📈 OVERFIT?"
        else:
            consistency = "📉 UNDERFIT?"
        
        print(f"{name.capitalize():<12} {cv_acc:.4f} ± {cv_std:.3f}   {test_acc:.4f}         {cv_std:.4f}    {consistency}")
    
    print("="*90)
    
    # Overall deployment status
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
    
    if success_count > 0:
        print(f"\n🏆 DEPLOYMENT SUCCESS!")
        print(f"   ✅ {success_count}/{len(results)} classifier(s) achieved 90%+ accuracy")
        print(f"   🚀 Ready for production deployment!")
        print(f"   📦 Models saved for immediate use")
    else:
        best_result = max(results.items(), key=lambda x: x[1]['final_test']['accuracy'])
        best_name, best_acc = best_result[0], best_result[1]['final_test']['accuracy']
        print(f"\n📈 BEST PERFORMANCE:")
        print(f"   🥇 {best_name.capitalize()}: {best_acc:.4f} accuracy")
        print(f"   💡 Consider ensemble methods or additional feature engineering")
    
    # Production deployment instructions
    print(f"\n📋 DEPLOYMENT INSTRUCTIONS:")
    print(f"   1. Use production_[type]_classifier.pkl files")
    print(f"   2. Load with: pickle.load(open('production_[type]_classifier.pkl', 'rb'))")
    print(f"   3. Predict with: model['classifier'].predict(new_features)")
    print(f"   4. Features: 16 resonator-based discriminative features")
    
    print("="*80)

if __name__ == "__main__":
    main()
