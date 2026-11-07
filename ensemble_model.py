#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ING Datathon - Ensemble Model Pipeline
LightGBM, XGBoost, CatBoost ile güçlü ensemble modeli
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Model imports
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.base import clone

# Yarışma metriği fonksiyonu
def recall_at_k(y_true, y_prob, k=0.1):
    """Tahmin edilen olasılıkların en üst k%'sını pozitif etiketleyerek recall değerini hesaplar."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    
    tp_at_k = y_true[top].sum()
    P = y_true.sum()
    
    return float(tp_at_k / P) if P > 0 else 0.0

def lift_at_k(y_true, y_prob, k=0.1):
    """Tahmin edilen olasılıkların en üst k%'sını pozitif etiketleyerek lift değerini hesaplar."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    
    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()
    
    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0

def convert_auc_to_gini(auc):
    """ROC AUC skorunu Gini katsayısına dönüştürür."""
    return 2 * auc - 1

def ing_hubs_datathon_metric(y_true, y_prob):
    """ING Datathon özel metriği"""
    score_weights = {
        "gini": 0.4,
        "recall_at_10perc": 0.3,
        "lift_at_10perc": 0.3,
    }
    
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }
    
    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)
    
    new_scores = {
        "roc_auc": roc_auc,
        "recall_at_10perc": recall_at_10perc,
        "lift_at_10perc": lift_at_10perc,
    }
    
    baseline_scores["gini"] = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_scores["gini"] = convert_auc_to_gini(new_scores["roc_auc"])
    
    final_gini_score = new_scores["gini"] / baseline_scores["gini"]
    final_recall_score = new_scores["recall_at_10perc"] / baseline_scores["recall_at_10perc"]
    final_lift_score = new_scores["lift_at_10perc"] / baseline_scores["lift_at_10perc"]
    
    final_score = (
        final_gini_score * score_weights["gini"] +
        final_recall_score * score_weights["recall_at_10perc"] + 
        final_lift_score * score_weights["lift_at_10perc"]
    )
    return final_score

class EnsembleModel:
    """Güçlü ensemble model sınıfı"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.cv_scores = {}
        
    def create_models(self):
        """Farklı algoritmalarla modeller oluşturur"""
        
        print("Modeller oluşturuluyor...")
        
        # LightGBM - Hızlı ve etkili
        self.models['lgb'] = LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=1000
        )
        
        # XGBoost - Güçlü performans
        self.models['xgb'] = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_estimators=1000,
            verbosity=0
        )
        
        # CatBoost - Kategorik özellikler için mükemmel
        self.models['cat'] = CatBoostClassifier(
            objective='Logloss',
            eval_metric='AUC',
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            iterations=1000,
            verbose=False
        )
        
        # Random Forest - Stabilite için
        self.models['rf'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees - Çeşitlilik için
        self.models['et'] = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"[OK] {len(self.models)} model oluşturuldu")
    
    def prepare_data(self, train_df, test_df):
        """Veriyi model eğitimi için hazırlar"""
        
        print("\nVeri hazırlanıyor...")
        
        # Kategorik sütunları belirle
        categorical_cols = ['gender', 'province', 'work_type', 'work_sector', 'age_group', 'tenure_group']
        
        # Label encoding
        label_encoders = {}
        for col in categorical_cols:
            if col in train_df.columns:
                le = LabelEncoder()
                # Train ve test verilerini birleştirerek encoding yap
                combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
                le.fit(combined.fillna('Unknown'))
                
                train_df[col] = le.transform(train_df[col].fillna('Unknown'))
                test_df[col] = le.transform(test_df[col].fillna('Unknown'))
                
                label_encoders[col] = le
        
        # Tarih kolonunu datetime formatına çevir
        if 'ref_date' in train_df.columns:
            train_df['ref_date'] = pd.to_datetime(train_df['ref_date'])
        if 'ref_date' in test_df.columns:
            test_df['ref_date'] = pd.to_datetime(test_df['ref_date'])
        
        # Feature sütunlarını belirle
        feature_cols = [col for col in train_df.columns 
                       if col not in ['cust_id', 'churn', 'is_train', 'ref_date']]
        
        print(f"[OK] {len(feature_cols)} özellik sütunu belirlendi")
        print(f"[OK] Kategorik sütunlar encode edildi")
        
        return train_df, test_df, feature_cols, label_encoders
    
    def build_time_based_folds(self, data, n_folds=4, val_span=2):
        """ref_date bazlı zaman serisi fold'ları üretir"""
        if 'ref_date' not in data.columns:
            raise ValueError("Time-based validation için 'ref_date' kolonuna ihtiyaç var.")
        
        ref_dates = np.array(sorted(pd.to_datetime(data['ref_date']).unique()))
        folds = []
        
        total_dates = len(ref_dates)
        if total_dates == 0:
            return folds
        
        for fold_idx in range(n_folds):
            start = total_dates - (fold_idx + 1) * val_span
            end = total_dates - fold_idx * val_span
            
            if start < 0:
                start = 0
            val_dates = ref_dates[start:end]
            if len(val_dates) == 0:
                continue
            
            train_mask = pd.to_datetime(data['ref_date']) < val_dates[0]
            val_mask = pd.to_datetime(data['ref_date']).isin(val_dates)
            
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue
            
            folds.append((data[train_mask].index, data[val_mask].index))
        
        folds.reverse()
        return folds
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Resmi metrik setini hesaplar"""
        auc = roc_auc_score(y_true, y_pred)
        gini = convert_auc_to_gini(auc)
        recall_10 = recall_at_k(y_true, y_pred, k=0.1)
        lift_10 = lift_at_k(y_true, y_pred, k=0.1)
        ing_score = ing_hubs_datathon_metric(y_true, y_pred)
        
        return {
            'auc': auc,
            'gini': gini,
            'recall_10': recall_10,
            'lift_10': lift_10,
            'ing_score': ing_score
        }
    
    def train_models(self, X_train, y_train, feature_cols):
        """Modelleri eğitir"""
        
        print("\nModeller eğitiliyor...")
        
        folds = self.build_time_based_folds(X_train)
        if not folds:
            raise ValueError("Zaman bazlı fold üretilemedi; ref_date kolonunu kontrol edin.")
        
        for name, model in self.models.items():
            print(f"\n{name.upper()} eğitiliyor...")
            
            fold_metrics = []
            
            for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
                fold_model = clone(model)
                
                X_tr = X_train.loc[train_idx, feature_cols]
                y_tr = y_train.loc[train_idx]
                X_val = X_train.loc[valid_idx, feature_cols]
                y_val = y_train.loc[valid_idx]
                
                fold_model.fit(X_tr, y_tr)
                val_pred = fold_model.predict_proba(X_val)[:, 1]
                
                metrics = self.calculate_metrics(y_val, val_pred)
                fold_metrics.append(metrics)
                
                print(
                    f"  Fold {fold_no}: ING={metrics['ing_score']:.4f} | "
                    f"AUC={metrics['auc']:.4f} | Recall@10={metrics['recall_10']:.4f} | "
                    f"Lift@10={metrics['lift_10']:.4f}"
                )
            
            avg_metrics = {
                'ing_score': np.mean([m['ing_score'] for m in fold_metrics]),
                'auc': np.mean([m['auc'] for m in fold_metrics]),
                'gini': np.mean([m['gini'] for m in fold_metrics]),
                'recall_10': np.mean([m['recall_10'] for m in fold_metrics]),
                'lift_10': np.mean([m['lift_10'] for m in fold_metrics]),
                'fold_details': fold_metrics
            }
            self.cv_scores[name] = avg_metrics
            
            print(
                f"  Ortalama: ING={avg_metrics['ing_score']:.4f} | "
                f"AUC={avg_metrics['auc']:.4f} | Recall@10={avg_metrics['recall_10']:.4f} | "
                f"Lift@10={avg_metrics['lift_10']:.4f}"
            )
            
            # Final modeli tüm eğitim verisi ile fit et
            model.fit(X_train[feature_cols], y_train)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
            elif hasattr(model, 'get_feature_importance'):
                self.feature_importance[name] = dict(zip(feature_cols, model.get_feature_importance()))
            
            print(f"  {name} eğitimi tamamlandı")
    
    def predict_ensemble(self, X_test, feature_cols, method='weighted'):
        """Ensemble tahminleri yapar"""
        
        print(f"\nEnsemble tahminleri yapılıyor ({method} method)...")
        
        predictions = {}
        
        # Her model için tahmin yap
        for name, model in self.models.items():
            pred = model.predict_proba(X_test[feature_cols])[:, 1]
            predictions[name] = pred
            print(f"  {name} tahminleri hazır")
        
        # Ensemble yöntemleri
        if method == 'weighted':
            # Zaman bazlı CV'den gelen ING skorlarına göre ağırlıklandırma
            weights = {}
            total_score = sum(self.cv_scores[name]['ing_score'] for name in self.cv_scores)
            
            for name in self.cv_scores:
                weights[name] = (
                    self.cv_scores[name]['ing_score'] / total_score if total_score > 0 else 1.0 / len(self.cv_scores)
                )
            
            # Ağırlıklandırılmış ortalama
            ensemble_pred = np.zeros(len(X_test))
            for name, pred in predictions.items():
                ensemble_pred += weights[name] * pred
                
        elif method == 'average':
            # Basit ortalama
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
        elif method == 'median':
            # Medyan
            ensemble_pred = np.median(list(predictions.values()), axis=0)
        
        print("Ensemble tahminleri hazır")
        
        return ensemble_pred, predictions
    
    def evaluate_model(self, y_true, y_pred):
        """Model performansını değerlendirir"""
        
        print("\nModel değerlendirmesi:")
        print("-" * 40)
        
        metrics = self.calculate_metrics(y_true, y_pred)
        
        print(f"ROC AUC: {metrics['auc']:.4f}")
        print(f"Gini: {metrics['gini']:.4f}")
        print(f"Recall@10%: {metrics['recall_10']:.4f}")
        print(f"Lift@10%: {metrics['lift_10']:.4f}")
        print(f"ING Datathon Score: {metrics['ing_score']:.4f}")
        
        return metrics
    
    def get_feature_importance_summary(self, feature_cols, top_n=20):
        """Feature importance özeti"""
        
        print(f"\nTop {top_n} Feature Importance:")
        print("-" * 50)
        
        # Tüm modellerden feature importance'ları topla
        all_importance = {}
        for col in feature_cols:
            all_importance[col] = 0
        
        for model_name, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                all_importance[feature] += importance
        
        # Ortalama al
        for feature in all_importance:
            all_importance[feature] /= len(self.feature_importance)
        
        # Sırala ve göster
        sorted_features = sorted(all_importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:top_n]):
            print(f"{i+1:2d}. {feature:<30} {importance:.4f}")

def main():
    """Ana fonksiyon"""
    
    print("=" * 60)
    print("ING DATATHON - ENSEMBLE MODEL PIPELINE")
    print("=" * 60)
    
    # Veri setlerini yükle
    print("\n1. Veri setleri yükleniyor...")
    train_df = pd.read_csv('train_features.csv')
    test_df = pd.read_csv('test_features.csv')
    
    print(f"[OK] Train: {train_df.shape}")
    print(f"[OK] Test: {test_df.shape}")
    
    # Ensemble model oluştur
    ensemble = EnsembleModel()
    ensemble.create_models()
    
    # Veriyi hazırla
    train_df, test_df, feature_cols, label_encoders = ensemble.prepare_data(train_df, test_df)
    
    # Train ve target ayır
    X_train = train_df.drop('churn', axis=1)
    y_train = train_df['churn']
    X_test = test_df.drop('churn', axis=1)
    
    print(f"\nTrain boyutu: {X_train.shape}")
    print(f"Test boyutu: {X_test.shape}")
    print(f"Churn oranı: {y_train.mean():.4f}")
    
    # Modelleri eğit
    ensemble.train_models(X_train, y_train, feature_cols)
    
    # Cross-validation skorları
    print("\nCross-Validation Skorları:")
    print("-" * 40)
    for name, scores in ensemble.cv_scores.items():
        print(
            f"{name.upper()}: ING={scores['ing_score']:.4f} | "
            f"AUC={scores['auc']:.4f} | Recall@10={scores['recall_10']:.4f} | "
            f"Lift@10={scores['lift_10']:.4f}"
        )
    
    # Feature importance
    ensemble.get_feature_importance_summary(feature_cols)
    
    # Test tahminleri
    test_predictions, individual_predictions = ensemble.predict_ensemble(X_test, feature_cols)
    
    # Submission dosyası oluştur
    submission = pd.DataFrame({
        'cust_id': test_df['cust_id'],
        'churn': test_predictions
    })
    
    submission.to_csv('ensemble_submission.csv', index=False)
    print(f"\n[OK] Submission dosyası 'ensemble_submission.csv' oluşturuldu!")
    
    # Individual model tahminleri de kaydet
    individual_submissions = {}
    for name, pred in individual_predictions.items():
        individual_submissions[name] = pd.DataFrame({
            'cust_id': test_df['cust_id'],
            'churn': pred
        })
        individual_submissions[name].to_csv(f'{name}_submission.csv', index=False)
        print(f"[OK] {name} submission dosyası oluşturuldu!")
    
    return ensemble, submission

if __name__ == "__main__":
    ensemble, submission = main()
