#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ING Datathon - Hyperparameter Optimization with Optuna
Optuna ile en iyi hyperparameter'ları bulur
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Model imports
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Yarışma metriği fonksiyonu
def ing_hubs_datathon_metric(y_true, y_prob):
    """ING Datathon özel metriği"""
    from sklearn.metrics import roc_auc_score
    
    def recall_at_k(y_true, y_prob, k=0.1):
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
        return 2 * auc - 1

    score_weights = {"gini": 0.4, "recall_at_10perc": 0.3, "lift_at_10perc": 0.3}
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }
    
    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)
    
    new_scores = {"roc_auc": roc_auc, "recall_at_10perc": recall_at_10perc, "lift_at_10perc": lift_at_10perc}
    
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

class HyperparameterOptimizer:
    """Optuna ile hyperparameter optimization"""
    
    def __init__(self, X_train, y_train, feature_cols):
        self.X_train = X_train.copy()
        self.y_train = y_train
        self.feature_cols = feature_cols
        self.best_params = {}
        self.ref_dates = pd.to_datetime(self.X_train['ref_date'])
        self.folds = self._build_time_based_folds()
        self.X_features = self.X_train[self.feature_cols]
    
    def _build_time_based_folds(self, n_folds=4, val_span=2):
        ref_dates = np.array(sorted(self.X_train['ref_date'].astype('datetime64[ns]').unique()))
        folds = []
        total_dates = len(ref_dates)
        
        for fold_idx in range(n_folds):
            start = total_dates - (fold_idx + 1) * val_span
            end = total_dates - fold_idx * val_span
            
            if start < 0:
                start = 0
            val_dates = ref_dates[start:end]
            if len(val_dates) == 0:
                continue
            
            train_mask = self.ref_dates < val_dates[0]
            val_mask = self.ref_dates.isin(val_dates)
            
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue
            
            folds.append((self.X_train[train_mask].index, self.X_train[val_mask].index))
        
        folds.reverse()
        if not folds:
            raise ValueError("Zaman bazlı fold oluşturulamadı; ref_date kolonunu kontrol edin.")
        return folds
        
    def optimize_lightgbm(self, n_trials=50):
        """LightGBM için hyperparameter optimization"""
        
        print("LightGBM hyperparameter optimization başlıyor...")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42,
                'verbose': -1,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000)
            }
            
            scores = []
            for train_idx, valid_idx in self.folds:
                model = LGBMClassifier(**params)
                model.fit(self.X_features.loc[train_idx], self.y_train.loc[train_idx])
                val_pred = model.predict_proba(self.X_features.loc[valid_idx])[:, 1]
                fold_score = ing_hubs_datathon_metric(self.y_train.loc[valid_idx], val_pred)
                scores.append(fold_score)
            
            return float(np.mean(scores))
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['lgb'] = study.best_params
        print(f"[OK] LightGBM en iyi skor: {study.best_value:.4f}")
        print(f"[OK] En iyi parametreler: {study.best_params}")
        
        return study.best_params
    
    def optimize_xgboost(self, n_trials=50):
        """XGBoost için hyperparameter optimization"""
        
        print("\nXGBoost hyperparameter optimization başlıyor...")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'verbosity': 0
            }
            
            scores = []
            for train_idx, valid_idx in self.folds:
                model = XGBClassifier(**params)
                model.fit(self.X_features.loc[train_idx], self.y_train.loc[train_idx])
                val_pred = model.predict_proba(self.X_features.loc[valid_idx])[:, 1]
                fold_score = ing_hubs_datathon_metric(self.y_train.loc[valid_idx], val_pred)
                scores.append(fold_score)
            
            return float(np.mean(scores))
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['xgb'] = study.best_params
        print(f"[OK] XGBoost en iyi skor: {study.best_value:.4f}")
        print(f"[OK] En iyi parametreler: {study.best_params}")
        
        return study.best_params
    
    def optimize_catboost(self, n_trials=30):
        """CatBoost için hyperparameter optimization"""
        
        print("\nCatBoost hyperparameter optimization başlıyor...")
        
        def objective(trial):
            params = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_seed': 42,
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'verbose': False
            }
            
            scores = []
            for train_idx, valid_idx in self.folds:
                model = CatBoostClassifier(**params)
                model.fit(self.X_features.loc[train_idx], self.y_train.loc[train_idx])
                val_pred = model.predict_proba(self.X_features.loc[valid_idx])[:, 1]
                fold_score = ing_hubs_datathon_metric(self.y_train.loc[valid_idx], val_pred)
                scores.append(fold_score)
            
            return float(np.mean(scores))
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['cat'] = study.best_params
        print(f"[OK] CatBoost en iyi skor: {study.best_value:.4f}")
        print(f"[OK] En iyi parametreler: {study.best_params}")
        
        return study.best_params
    
    def optimize_all_models(self, n_trials_per_model=30):
        """Tüm modeller için hyperparameter optimization"""
        
        print("=" * 60)
        print("HYPERPARAMETER OPTIMIZATION BAŞLIYOR")
        print("=" * 60)
        
        # LightGBM optimization
        lgb_params = self.optimize_lightgbm(n_trials_per_model)
        
        # XGBoost optimization
        xgb_params = self.optimize_xgboost(n_trials_per_model)
        
        # CatBoost optimization
        cat_params = self.optimize_catboost(n_trials_per_model)
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION TAMAMLANDI!")
        print("=" * 60)
        
        return self.best_params

def create_optimized_ensemble():
    """Optimize edilmiş ensemble model oluşturur"""
    
    print("Optimize edilmiş ensemble model oluşturuluyor...")
    
    # Veri setlerini yükle
    train_df = pd.read_csv('train_features.csv')
    test_df = pd.read_csv('test_features.csv')
    
    if 'ref_date' in train_df.columns:
        train_df['ref_date'] = pd.to_datetime(train_df['ref_date'])
    if 'ref_date' in test_df.columns:
        test_df['ref_date'] = pd.to_datetime(test_df['ref_date'])
    
    # Kategorik sütunları encode et
    from sklearn.preprocessing import LabelEncoder
    
    categorical_cols = ['gender', 'province', 'work_type', 'work_sector', 'age_group', 'tenure_group']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
            le.fit(combined.fillna('Unknown'))
            
            train_df[col] = le.transform(train_df[col].fillna('Unknown'))
            test_df[col] = le.transform(test_df[col].fillna('Unknown'))
            
            label_encoders[col] = le
    
    # Feature sütunlarını belirle
    feature_cols = [col for col in train_df.columns 
                   if col not in ['cust_id', 'churn', 'is_train', 'ref_date']]
    
    # Train ve target ayır
    X_train = train_df.drop('churn', axis=1)
    y_train = train_df['churn']
    X_test = test_df.drop('churn', axis=1)
    
    # Hyperparameter optimization
    optimizer = HyperparameterOptimizer(X_train, y_train, feature_cols)
    best_params = optimizer.optimize_all_models(n_trials_per_model=20)  # Hızlı test için
    
    # Optimize edilmiş modelleri oluştur
    print("\nOptimize edilmiş modeller eğitiliyor...")
    
    models = {}
    
    # LightGBM
    lgb_model = LGBMClassifier(**best_params['lgb'])
    lgb_model.fit(X_train[feature_cols], y_train)
    models['lgb'] = lgb_model
    
    # XGBoost
    xgb_model = XGBClassifier(**best_params['xgb'])
    xgb_model.fit(X_train[feature_cols], y_train)
    models['xgb'] = xgb_model
    
    # CatBoost
    cat_model = CatBoostClassifier(**best_params['cat'])
    cat_model.fit(X_train[feature_cols], y_train)
    models['cat'] = cat_model
    
    # Test tahminleri
    print("\nTest tahminleri yapılıyor...")
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict_proba(X_test[feature_cols])[:, 1]
        predictions[name] = pred
        print(f"[OK] {name} tahminleri hazır")
    
    # Ensemble tahminleri (ağırlıklandırılmış ortalama)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    # Submission dosyası
    submission = pd.DataFrame({
        'cust_id': test_df['cust_id'],
        'churn': ensemble_pred
    })
    
    submission.to_csv('optimized_ensemble_submission.csv', index=False)
    print(f"\n[OK] Optimize edilmiş submission dosyası oluşturuldu!")
    
    # Individual model submissions
    for name, pred in predictions.items():
        individual_submission = pd.DataFrame({
            'cust_id': test_df['cust_id'],
            'churn': pred
        })
        individual_submission.to_csv(f'optimized_{name}_submission.csv', index=False)
        print(f"[OK] Optimize edilmiş {name} submission dosyası oluşturuldu!")
    
    return models, submission, best_params

def main():
    """Ana fonksiyon"""
    
    print("=" * 60)
    print("ING DATATHON - HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    models, submission, best_params = create_optimized_ensemble()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION TAMAMLANDI!")
    print("=" * 60)
    
    return models, submission, best_params

if __name__ == "__main__":
    models, submission, best_params = main()
