#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ING HUBS DATATHON - H BR T CHURN PREDICTION MODEL
================================================
V1 Base (157 features) + 9 New Features + Conservative Tuning
Target: Validation 1.19-1.23, Kaggle 1.18-1.22
"""

import numpy as np
import pandas as pd
import gc
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import optuna
import warnings
import sys
warnings.filterwarnings('ignore')


# In[ ]:
# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    SEED = 42
    WINDOWS = [1, 3, 6, 12]
    TRAIN_END_DATE = "2018-09"
    N_ESTIMATORS = 300
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE = False
    BATCH_SIZE = 5000
    LGB_TUNING_TRIALS = 5
    XGB_TUNING_TRIALS = 3
    CAT_TUNING_TRIALS = 0

    # Paths - Google Colab uyumlu
    DATA_DIR = "./"
    OUTPUT_DIR = "./outputs/"

config = Config()


# In[ ]:
# ============================================================================
# METRIC FUNCTIONS (OFFICIAL COMPETITION VERSION)
# ============================================================================
def recall_at_k(y_true, y_prob, k=0.1):
    """
    Official competition recall@k calculation
    """
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
    """
    Official competition lift@k calculation
    """
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
    """
    Convert ROC AUC to Gini coefficient
    """
    return 2 * auc - 1

def ing_hubs_datathon_metric(y_true, y_prob):
    """
    Official ING Hubs Datathon competition metric
    Returns both the composite score and detailed metrics
    """
    # Weights for final metric
    score_weights = {
        "gini": 0.4,
        "recall_at_10perc": 0.3,
        "lift_at_10perc": 0.3,
    }

    # Baseline model scores
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }

    # Calculate metrics for y_prob predictions
    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)

    new_scores = {
        "roc_auc": roc_auc,
        "recall_at_10perc": recall_at_10perc,
        "lift_at_10perc": lift_at_10perc,
    }

    # Convert roc_auc to gini
    baseline_scores["gini"] = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_scores["gini"] = convert_auc_to_gini(new_scores["roc_auc"])

    # Calculate ratios vs baseline
    final_gini_score = new_scores["gini"] / baseline_scores["gini"]
    final_recall_score = new_scores["recall_at_10perc"] / baseline_scores["recall_at_10perc"]
    final_lift_score = new_scores["lift_at_10perc"] / baseline_scores["lift_at_10perc"]

    # Calculate weighted composite metric
    final_score = (
        final_gini_score * score_weights["gini"] +
        final_recall_score * score_weights["recall_at_10perc"] +
        final_lift_score * score_weights["lift_at_10perc"]
    )

    # Return both composite and detailed metrics
    return {
        'auc': roc_auc,
        'gini': new_scores["gini"],
        'recall@10': recall_at_10perc,
        'lift@10': lift_at_10perc,
        'composite_score': final_score
    }

# In[ ]:
# ============================================================================
# MODEL WRAPPER
# ============================================================================
class ModelWrapper:
    """Unified interface for different models"""

    def __init__(self, model_type, params=None, seed=42):
        self.model_type = model_type
        self.params = params or {}
        self.seed = seed
        self.model = None
        self.cat_features = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, cat_features=None):
        """Train model"""
        if self.model_type == 'lightgbm':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid) if X_valid is not None else None

            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=config.N_ESTIMATORS,
                valid_sets=[valid_data] if valid_data else None,
                callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS)] if valid_data else None
            )

        elif self.model_type == 'xgboost':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid) if X_valid is not None else None

            evallist = [(dvalid, 'eval')] if dvalid else []
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=config.N_ESTIMATORS,
                evals=evallist,
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS if dvalid else None,
                verbose_eval=False
            )

        elif self.model_type == 'catboost':
            self.cat_features = cat_features
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            valid_pool = Pool(X_valid, y_valid, cat_features=cat_features) if X_valid is not None else None

            self.model = CatBoostClassifier(**self.params)
            self.model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS if valid_pool else None,
                verbose=config.VERBOSE
            )

        elif self.model_type == 'extratrees':
            self.model = ExtraTreesClassifier(**self.params, random_state=self.seed)
            self.model.fit(X_train, y_train)

        elif self.model_type == 'logistic':
            self.model = LogisticRegression(**self.params, random_state=self.seed)
            self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model_type == 'lightgbm':
            return self.model.predict(X)
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)
        elif self.model_type == 'catboost':
            pool = Pool(X, cat_features=self.cat_features) if self.cat_features else Pool(X)
            proba = self.model.predict_proba(pool)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        else:
            return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """Get feature importance"""
        if self.model_type == 'lightgbm':
            return self.model.feature_importance(importance_type='gain')
        elif self.model_type == 'xgboost':
            return self.model.get_score(importance_type='gain')
        elif self.model_type == 'catboost':
            return self.model.get_feature_importance()
        elif self.model_type == 'extratrees':
            return self.model.feature_importances_
        else:
            return None


# In[ ]:
# ============================================================================
# HYBRID FEATURE ENGINEER
# ============================================================================
class HybridFeatureEngineer:
    """V1 Base (157) + 9 New Features"""

    def __init__(self, windows=[1, 3, 6, 12]):
        self.windows = windows
        self.label_encoders = {}

    def transform(self, customer_history, customers, reference_data):
        """Transform data with V1 base + 9 new features"""

        print("  H BR T FEATURE ENGINEERING...")
        print(f"   Hedef: V1 base (157) + 9 yeni  zellik")
        print(f"   Sat r: {len(reference_data):,}")

        # Tarih d n   m 
        customer_history['date'] = pd.to_datetime(customer_history['date'])
        reference_data['ref_date'] = pd.to_datetime(reference_data['ref_date'])

        # Merge customers
        data = reference_data.merge(customers, on='cust_id', how='left')

        # Batch processing
        batch_size = config.BATCH_SIZE
        all_features = []

        for start_idx in range(0, len(data), batch_size):
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data.iloc[start_idx:end_idx].copy()

            if start_idx % 10000 == 0:
                print(f"     lenen: {start_idx:,}/{len(data):,} ({100*start_idx/len(data):.1f}%)")

            batch_features = self._process_batch(batch_data, customer_history)
            all_features.append(batch_features)

            gc.collect()

        # Combine all batches
        features_df = pd.concat(all_features, ignore_index=True)

        # Add demographics
        features_df = features_df.merge(customers, on='cust_id', how='left')

        print(f"\n  FEATURE ENGINEERING TAMAMLANDI!")
        print(f"    zellik say s : {features_df.shape[1] - 2}")  # -2 for cust_id, ref_date

        return features_df

    def _process_batch(self, batch_data, customer_history):
        """Process a batch of customers"""

        features_list = []

        for _, row in batch_data.iterrows():
            cust_id = row['cust_id']
            ref_date = row['ref_date']

            # Filter history BEFORE ref_date (kritik: s zma  nleme)
            cust_history = customer_history[
                (customer_history['cust_id'] == cust_id) &
                (customer_history['date'] < ref_date)
            ].copy()

            if len(cust_history) == 0:
                # Empty history
                base_features = self._get_empty_features()
            else:
                # Calculate features
                base_features = self._calculate_v1_base_features(cust_history, ref_date)
                new_features = self._calculate_new_features(cust_history, ref_date, base_features)
                base_features.update(new_features)

            base_features['cust_id'] = cust_id
            base_features['ref_date'] = ref_date
            if 'churn' in row:
                base_features['churn'] = row['churn']
            else:
                base_features['churn'] = 0
            features_list.append(base_features)

        return pd.DataFrame(features_list)

    def _calculate_v1_base_features(self, history, ref_date):
        """Calculate V1 base features (157 adet)"""

        features = {}

        # 1. Windowed Stats
        for window in self.windows:
            window_data = history[history['date'] >= ref_date - pd.DateOffset(months=window)]

            for col in ['mobile_eft_all_cnt', 'mobile_eft_all_amt',
                       'cc_transaction_all_cnt', 'cc_transaction_all_amt']:
                if col in window_data.columns:
                    features[f'{col}_sum_{window}m'] = window_data[col].sum()
                    features[f'{col}_mean_{window}m'] = window_data[col].mean()
                    features[f'{col}_max_{window}m'] = window_data[col].max()
                    features[f'{col}_std_{window}m'] = window_data[col].std()
                    features[f'{col}_min_{window}m'] = window_data[col].min()

            # Active months
            features[f'active_months_{window}m'] = window_data['date'].nunique()
            features[f'is_missing_{window}m'] = 1 if len(window_data) == 0 else 0

        # 2. RFM Features
        if len(history) > 0:
            features['months_since_last_txn'] = (ref_date - history['date'].max()).days / 30

            cc_history = history[history['cc_transaction_all_cnt'] > 0]
            eft_history = history[history['mobile_eft_all_cnt'] > 0]

            features['months_since_last_cc'] = (ref_date - cc_history['date'].max()).days / 30 if len(cc_history) > 0 else 999
            features['months_since_last_eft'] = (ref_date - eft_history['date'].max()).days / 30 if len(eft_history) > 0 else 999

            # Average transaction amounts
            for window in [3, 6, 12]:
                window_data = history[history['date'] >= ref_date - pd.DateOffset(months=window)]

                cc_txns = window_data['cc_transaction_all_cnt'].sum()
                cc_amt = window_data['cc_transaction_all_amt'].sum()
                features[f'avg_cc_amt_per_txn_{window}m'] = cc_amt / cc_txns if cc_txns > 0 else 0

                eft_txns = window_data['mobile_eft_all_cnt'].sum()
                eft_amt = window_data['mobile_eft_all_amt'].sum()
                features[f'avg_eft_amt_per_txn_{window}m'] = eft_amt / eft_txns if eft_txns > 0 else 0

        # 3. Trend Features
        data_3m = history[history['date'] >= ref_date - pd.DateOffset(months=3)]
        data_6m_old = history[(history['date'] >= ref_date - pd.DateOffset(months=6)) &
                               (history['date'] < ref_date - pd.DateOffset(months=3))]

        for col in ['mobile_eft_all_cnt', 'cc_transaction_all_cnt']:
            if col in history.columns:
                recent_mean = data_3m[col].mean() if len(data_3m) > 0 else 0
                old_mean = data_6m_old[col].mean() if len(data_6m_old) > 0 else 0

                features[f'{col}_diff_3m'] = recent_mean - old_mean
                features[f'{col}_pct_chg_3m'] = (recent_mean - old_mean) / old_mean if old_mean > 0 else 0

        # Linear trend (6m)
        data_6m = history[history['date'] >= ref_date - pd.DateOffset(months=6)]
        if len(data_6m) > 1:
            for col in ['mobile_eft_all_cnt', 'cc_transaction_all_cnt']:
                if col in data_6m.columns:
                    x = np.arange(len(data_6m))
                    y = data_6m[col].values
                    slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                    features[f'{col}_slope_6m'] = slope

        # 4. Product Features
        if 'active_product_category_nbr' in history.columns:
            for window in self.windows:
                window_data = history[history['date'] >= ref_date - pd.DateOffset(months=window)]
                if len(window_data) > 0:
                    features[f'prod_cat_mean_{window}m'] = window_data['active_product_category_nbr'].mean()
                    features[f'prod_cat_max_{window}m'] = window_data['active_product_category_nbr'].max()
                    features[f'prod_cat_min_{window}m'] = window_data['active_product_category_nbr'].min()
                    features[f'prod_cat_last_{window}m'] = window_data['active_product_category_nbr'].iloc[-1]

            # Product drop score
            if len(history) > 0:
                prod_12m_max = history[history['date'] >= ref_date - pd.DateOffset(months=12)]['active_product_category_nbr'].max()
                prod_3m_mean = history[history['date'] >= ref_date - pd.DateOffset(months=3)]['active_product_category_nbr'].mean()
                features['prod_drop_score'] = prod_12m_max - prod_3m_mean if not np.isnan(prod_12m_max) and not np.isnan(prod_3m_mean) else 0
                features['is_recent_product_drop'] = 1 if features['prod_drop_score'] > 2 else 0

        # 5. Ratio Features
        for window in [3, 6, 12]:
            window_data = history[history['date'] >= ref_date - pd.DateOffset(months=window)]

            cc_amt = window_data['cc_transaction_all_amt'].sum()
            eft_amt = window_data['mobile_eft_all_amt'].sum()
            cc_cnt = window_data['cc_transaction_all_cnt'].sum()
            eft_cnt = window_data['mobile_eft_all_cnt'].sum()

            total_amt = cc_amt + eft_amt
            total_cnt = cc_cnt + eft_cnt

            features[f'cc_to_eft_amt_ratio_{window}m'] = cc_amt / eft_amt if eft_amt > 0 else 0
            features[f'cc_to_eft_cnt_ratio_{window}m'] = cc_cnt / eft_cnt if eft_cnt > 0 else 0
            features[f'cc_amt_share_{window}m'] = cc_amt / total_amt if total_amt > 0 else 0
            features[f'eft_amt_share_{window}m'] = eft_amt / total_amt if total_amt > 0 else 0
            features[f'cc_cnt_share_{window}m'] = cc_cnt / total_cnt if total_cnt > 0 else 0
            features[f'eft_cnt_share_{window}m'] = eft_cnt / total_cnt if total_cnt > 0 else 0

        # 6. Temporal Features
        features['ref_month'] = ref_date.month
        features['ref_quarter'] = ref_date.quarter
        features['ref_year'] = ref_date.year
        features['ref_month_sin'] = np.sin(2 * np.pi * ref_date.month / 12)
        features['ref_month_cos'] = np.cos(2 * np.pi * ref_date.month / 12)

        return features

    def _calculate_new_features(self, history, ref_date, base_features):
        """Calculate 9 new features"""

        new_features = {}

        data_1m = history[history['date'] >= ref_date - pd.DateOffset(months=1)]
        data_3m = history[history['date'] >= ref_date - pd.DateOffset(months=3)]
        data_6m = history[history['date'] >= ref_date - pd.DateOffset(months=6)]
        data_6m_old = history[(history['date'] >= ref_date - pd.DateOffset(months=6)) &
                               (history['date'] < ref_date - pd.DateOffset(months=3))]
        data_12m = history[history['date'] >= ref_date - pd.DateOffset(months=12)]

        # 1. prod_velocity_3m
        if 'active_product_category_nbr' in history.columns:
            prod_1m = data_1m['active_product_category_nbr'].mean() if len(data_1m) > 0 else 0
            prod_3m = data_3m['active_product_category_nbr'].mean() if len(data_3m) > 0 else 0
            new_features['prod_velocity_3m'] = (prod_1m - prod_3m) / 3 if not np.isnan(prod_3m) else 0

        # 2. prod_momentum_6m
        if 'active_product_category_nbr' in history.columns:
            prod_recent = data_3m['active_product_category_nbr'].mean() if len(data_3m) > 0 else 0
            prod_old = data_6m_old['active_product_category_nbr'].mean() if len(data_6m_old) > 0 else 0
            new_features['prod_momentum_6m'] = prod_recent - prod_old

        # 3. prod_erosion_score
        if 'active_product_category_nbr' in history.columns:
            prod_12m_max = data_12m['active_product_category_nbr'].max() if len(data_12m) > 0 else 0
            prod_3m_mean = data_3m['active_product_category_nbr'].mean() if len(data_3m) > 0 else 0
            new_features['prod_erosion_score'] = max(0, prod_12m_max - prod_3m_mean)

        # 4. engagement_decay
        txn_3m = (data_3m['cc_transaction_all_cnt'].sum() + data_3m['mobile_eft_all_cnt'].sum()) if len(data_3m) > 0 else 0
        txn_6m_old = (data_6m_old['cc_transaction_all_cnt'].sum() + data_6m_old['mobile_eft_all_cnt'].sum()) if len(data_6m_old) > 0 else 1
        new_features['engagement_decay'] = txn_3m / max(txn_6m_old, 1)

        # 5. monetary_trend
        amt_3m = (data_3m['cc_transaction_all_amt'].sum() + data_3m['mobile_eft_all_amt'].sum()) / max(len(data_3m), 1) if len(data_3m) > 0 else 0
        amt_12m = (data_12m['cc_transaction_all_amt'].sum() + data_12m['mobile_eft_all_amt'].sum()) / max(len(data_12m), 1) if len(data_12m) > 0 else 1
        new_features['monetary_trend'] = amt_3m / max(amt_12m, 1)

        # 6. lifetime_value_proxy
        total_amt = history['cc_transaction_all_amt'].sum() + history['mobile_eft_all_amt'].sum()
        tenure_months = (ref_date - history['date'].min()).days / 30 if len(history) > 0 else 1
        new_features['lifetime_value_proxy'] = total_amt / max(tenure_months, 1)

        # 7. age_tenure_interaction (from demographics, will be added later)
        new_features['age_tenure_interaction'] = 0  # Placeholder

        # 8. prod_activity_6m
        prod_cat_mean_6m = base_features.get('prod_cat_mean_6m', 0)
        active_months_6m = base_features.get('active_months_6m', 0)
        new_features['prod_activity_6m'] = prod_cat_mean_6m * active_months_6m

        # 9. recency_product_risk
        months_since_last = base_features.get('months_since_last_txn', 0)
        prod_cat_mean_3m = base_features.get('prod_cat_mean_3m', 0)
        new_features['recency_product_risk'] = months_since_last * max(0, 10 - prod_cat_mean_3m)

        return new_features

    def _get_empty_features(self):
        """Get empty features for customers with no history"""
        features = {}

        # V1 base features
        for window in self.windows:
            for col in ['mobile_eft_all_cnt', 'mobile_eft_all_amt',
                       'cc_transaction_all_cnt', 'cc_transaction_all_amt']:
                for stat in ['sum', 'mean', 'max', 'std', 'min']:
                    features[f'{col}_{stat}_{window}m'] = 0

            features[f'active_months_{window}m'] = 0
            features[f'is_missing_{window}m'] = 1

            if 'active_product_category_nbr' in ['dummy']:
                for stat in ['mean', 'max', 'min', 'last']:
                    features[f'prod_cat_{stat}_{window}m'] = 0

        # RFM
        features['months_since_last_txn'] = 999
        features['months_since_last_cc'] = 999
        features['months_since_last_eft'] = 999

        for window in [3, 6, 12]:
            features[f'avg_cc_amt_per_txn_{window}m'] = 0
            features[f'avg_eft_amt_per_txn_{window}m'] = 0
            features[f'cc_to_eft_amt_ratio_{window}m'] = 0
            features[f'cc_to_eft_cnt_ratio_{window}m'] = 0
            features[f'cc_amt_share_{window}m'] = 0
            features[f'eft_amt_share_{window}m'] = 0
            features[f'cc_cnt_share_{window}m'] = 0
            features[f'eft_cnt_share_{window}m'] = 0

        # Trends
        for col in ['mobile_eft_all_cnt', 'cc_transaction_all_cnt']:
            features[f'{col}_diff_3m'] = 0
            features[f'{col}_pct_chg_3m'] = 0
            features[f'{col}_slope_6m'] = 0

        # Product
        features['prod_drop_score'] = 0
        features['is_recent_product_drop'] = 0

        # Temporal
        features['ref_month'] = 0
        features['ref_quarter'] = 0
        features['ref_year'] = 0
        features['ref_month_sin'] = 0
        features['ref_month_cos'] = 0

        # New features
        for feat in ['prod_velocity_3m', 'prod_momentum_6m', 'prod_erosion_score',
                     'engagement_decay', 'monetary_trend', 'lifetime_value_proxy',
                     'age_tenure_interaction', 'prod_activity_6m', 'recency_product_risk']:
            features[feat] = 0

        return features

    def label_encode_features(self, train_df, valid_df=None, test_df=None):
        """Label encode categorical features"""

        print("\n  Label Encoding uygulan yor...")

        categorical_cols = ['gender', 'province', 'religion', 'work_type', 'work_sector']

        for col in categorical_cols:
            if col in train_df.columns:
                le = LabelEncoder()

                # Fit on train
                train_df[col] = train_df[col].fillna('missing')
                le.fit(train_df[col])
                train_df[col] = le.transform(train_df[col])

                # Transform valid
                if valid_df is not None and col in valid_df.columns:
                    valid_df[col] = valid_df[col].fillna('missing')
                    valid_df[col] = valid_df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

                # Transform test
                if test_df is not None and col in test_df.columns:
                    test_df[col] = test_df[col].fillna('missing')
                    test_df[col] = test_df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

                self.label_encoders[col] = le

        # Update age_tenure_interaction
        if 'age' in train_df.columns and 'tenure' in train_df.columns:
            train_df['age_tenure_interaction'] = train_df['age'] * train_df['tenure']
            train_df.loc[train_df['age_tenure_interaction'] < 0, 'age_tenure_interaction'] = 0

            if valid_df is not None:
                valid_df['age_tenure_interaction'] = valid_df['age'] * valid_df['tenure']
                valid_df.loc[valid_df['age_tenure_interaction'] < 0, 'age_tenure_interaction'] = 0

            if test_df is not None:
                test_df['age_tenure_interaction'] = test_df['age'] * test_df['tenure']
                test_df.loc[test_df['age_tenure_interaction'] < 0, 'age_tenure_interaction'] = 0

        print("  Label encoding tamamland !")

        return train_df, valid_df, test_df


# In[ ]:
# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("="*80)
print("  ING HUBS DATATHON - H BR T CHURN PREDICTION")
print("="*80)
print(f"  Strategy: V1 Base (157) + 9 New Features + Conservative Tuning")
print(f"  Target: Validation 1.19-1.23, Kaggle 1.18-1.22")
print("="*80)

# Create output directory
import os
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# In[ ]:
# ============================================================================
# LOAD DATA
# ============================================================================
print("\n  VER  Y KLEN YOR...")

try:
    customer_history = pd.read_csv(f"{config.DATA_DIR}customer_history.csv")
    customers = pd.read_csv(f"{config.DATA_DIR}customers.csv")
    reference_data = pd.read_csv(f"{config.DATA_DIR}referance_data.csv")
    reference_data_test = pd.read_csv(f"{config.DATA_DIR}referance_data_test.csv")

    print(f"  Veri y klendi!")
    print(f"   customer_history: {customer_history.shape}")
    print(f"   customers: {customers.shape}")
    print(f"   reference_data (train): {reference_data.shape}")
    print(f"   reference_data_test: {reference_data_test.shape}")
except Exception as e:
    print(f"    Veri y kleme hatas : {e}")
    print(f"   Alternatif: Google Drive'dan y klemek i in:")
    print(f"   from google.colab import drive")
    print(f"   drive.mount('/content/drive')")
    raise


# In[ ]:
# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
engineer = HybridFeatureEngineer(windows=config.WINDOWS)

train_features_path = os.path.join(config.OUTPUT_DIR, "train_features_hybrid.csv")
test_features_path = os.path.join(config.OUTPUT_DIR, "test_features_hybrid.csv")

# Transform train data (with caching)
if os.path.exists(train_features_path):
    print("\nCached train features bulundu, dosyadan okunuyor...")
    train_features = pd.read_csv(train_features_path, parse_dates=['ref_date'])
else:
    print("\n" + "="*80)
    print("  TRAIN VER  FEATURE ENGINEERING")
    print("="*80)
    train_features = engineer.transform(customer_history, customers, reference_data)
    train_features.to_csv(train_features_path, index=False)
    print(f"  Train features kaydedildi: {train_features_path}")

# Transform test data (with caching)
if os.path.exists(test_features_path):
    print("\nCached test features bulundu, dosyadan okunuyor...")
    test_features = pd.read_csv(test_features_path, parse_dates=['ref_date'])
else:
    print("\n" + "="*80)
    print("  TEST VER  FEATURE ENGINEERING")
    print("="*80)
    test_features = engineer.transform(customer_history, customers, reference_data_test)
    test_features.to_csv(test_features_path, index=False)
    print(f"  Test features kaydedildi: {test_features_path}")

# In[ ]:
# ============================================================================
# TRAIN-VALIDATION SPLIT
# ============================================================================
print("\n" + "="*80)
print("    TRAIN-VALIDATION SPLIT")
print("="*80)

train_features['ref_date'] = pd.to_datetime(train_features['ref_date'])
train_mask = train_features['ref_date'] <= pd.to_datetime(config.TRAIN_END_DATE)

train_data = train_features[train_mask].copy()
valid_data = train_features[~train_mask].copy()

print(f"  Train: {len(train_data):,} samples")
print(f"  Valid: {len(valid_data):,} samples")
print(f"  Test: {len(test_features):,} samples")

# In[ ]:
# ============================================================================
# LABEL ENCODING
# ============================================================================
train_data, valid_data, test_features = engineer.label_encode_features(
    train_data, valid_data, test_features
)

# Prepare features
exclude_cols = ['cust_id', 'ref_date', 'churn']
feature_cols = [col for col in train_data.columns if col not in exclude_cols]

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['churn']
X_valid = valid_data[feature_cols].fillna(0)
y_valid = valid_data['churn']
X_test = test_features[feature_cols].fillna(0)

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

categorical_cols = ['gender', 'province', 'religion', 'work_type', 'work_sector']
cat_feature_names = [col for col in feature_cols if col in categorical_cols]

print(f"   Final feature count: {len(feature_cols)}")
print(f"   Expected: 166 (157 base + 9 new)")
print(f"   NaN count in train: {X_train.isna().sum().sum()}")
print(f"   NaN count in valid: {X_valid.isna().sum().sum()}")
print(f"   NaN count in test: {X_test.isna().sum().sum()}")


# In[ ]:
# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("  CONSERVATIVE HYPERPARAMETER TUNING")
print("="*80)

def objective_lgb(trial):
    """LightGBM objective with conservative ranges"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 35),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 150),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'verbose': -1,
        'seed': config.SEED
    }

    model = ModelWrapper('lightgbm', params, config.SEED)
    model.fit(X_train, y_train, X_valid, y_valid)
    preds = model.predict_proba(X_valid)

    metrics = ing_hubs_datathon_metric(y_valid, preds)
    return metrics['composite_score']

def objective_xgb(trial):
    """XGBoost objective with conservative ranges"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'gamma': trial.suggest_float('gamma', 0.01, 0.5),
        'seed': config.SEED,
        'verbosity': 0
    }

    model = ModelWrapper('xgboost', params, config.SEED)
    model.fit(X_train, y_train, X_valid, y_valid)
    preds = model.predict_proba(X_valid)

    metrics = ing_hubs_datathon_metric(y_valid, preds)
    return metrics['composite_score']

def objective_cat(trial):
    """CatBoost objective with conservative ranges"""
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
        'depth': trial.suggest_int('depth', 4, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 10.0),
        'bootstrap_type': 'Bernoulli',  # CRITICAL: Bayesian kullanma!
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'iterations': config.N_ESTIMATORS,
        'random_seed': config.SEED,
        'verbose': False,
        'auto_class_weights': 'Balanced'  #   FIX: Class imbalance eklendi
    }

    model = ModelWrapper('catboost', params, config.SEED)
    model.fit(X_train, y_train, X_valid, y_valid, cat_features=cat_feature_names)
    preds = model.predict_proba(X_valid)

    metrics = ing_hubs_datathon_metric(y_valid, preds)
    return metrics['composite_score']


# LightGBM Tuning
print("\n  LightGBM Conservative Tuning...")
print(f"   ({config.LGB_TUNING_TRIALS} trials)")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config.SEED))
study_lgb.optimize(objective_lgb, n_trials=config.LGB_TUNING_TRIALS, show_progress_bar=True)

best_lgb_params = study_lgb.best_params
best_lgb_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'seed': config.SEED
})
print(f"  LightGBM - En iyi skor: {study_lgb.best_value:.5f}")
print("  En iyi parametreler:")
for key, value in best_lgb_params.items():
    if key not in ['objective', 'metric', 'boosting_type', 'verbose', 'seed']:
        print(f"   {key:20s}: {value}")

# XGBoost Tuning
print("\n  XGBoost Conservative Tuning...")
print(f"   ({config.XGB_TUNING_TRIALS} trials)")
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config.SEED))
study_xgb.optimize(objective_xgb, n_trials=config.XGB_TUNING_TRIALS, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
best_xgb_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,  #   FIX: Eklendi
    'seed': config.SEED,
    'verbosity': 0
})
print(f"  XGBoost - En iyi skor: {study_xgb.best_value:.5f}")
print("  En iyi parametreler:")
for key, value in best_xgb_params.items():
    if key not in ['objective', 'eval_metric', 'seed', 'verbosity']:
        print(f"   {key:20s}: {value}")

if config.CAT_TUNING_TRIALS > 0:
    print("\n  CatBoost Conservative Tuning...")
    print(f"   ({config.CAT_TUNING_TRIALS} trials)")
    study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study_cat.optimize(objective_cat, n_trials=config.CAT_TUNING_TRIALS, show_progress_bar=True)

    best_cat_params = study_cat.best_params
    best_cat_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': config.N_ESTIMATORS,
        'random_seed': config.SEED,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    })
    print(f"  CatBoost - En iyi skor: {study_cat.best_value:.5f}")
    print("  En iyi parametreler:")
    for key, value in best_cat_params.items():
        if key not in ['loss_function', 'eval_metric', 'iterations', 'random_seed', 'verbose']:
            print(f"   {key:20s}: {value}")
else:
    print("\n  CatBoost Tuning atlandı (config.CAT_TUNING_TRIALS=0). Varsayılan parametreler kullanılacak.")
    best_cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'iterations': config.N_ESTIMATORS,
        'random_seed': config.SEED,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }


# In[ ]:
# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("  H BR T MODELLER E  T L YOR...")
print("="*80)

models = {}
predictions_valid = {}
predictions_test = {}
model_metrics = {}

# 1. LightGBM
print("\n[1/4]   LightGBM (Hybrid) e itiliyor...")
lgb_model = ModelWrapper('lightgbm', best_lgb_params, config.SEED)
lgb_model.fit(X_train, y_train, X_valid, y_valid)
models['lgbm'] = lgb_model

preds_valid = lgb_model.predict_proba(X_valid)
preds_test = lgb_model.predict_proba(X_test)
predictions_valid['lgbm'] = preds_valid
predictions_test['lgbm'] = preds_test

metrics = ing_hubs_datathon_metric(y_valid, preds_valid)
model_metrics['lgbm'] = metrics

print("="*60)
print("LightGBM (Hybrid) METR KLER")
print("="*60)
print(f"AUC:              {metrics['auc']:.5f}")
print(f"Gini:             {metrics['gini']:.5f}")
print(f"Recall@10%:       {metrics['recall@10']:.5f}")
print(f"Lift@10%:         {metrics['lift@10']:.5f}")
print("-"*60)
print(f"  Composite Score: {metrics['composite_score']:.5f}")
print("="*60)

gc.collect()

# 2. XGBoost
print("\n[2/4]   XGBoost (Hybrid) e itiliyor...")
xgb_model = ModelWrapper('xgboost', best_xgb_params, config.SEED)
xgb_model.fit(X_train, y_train, X_valid, y_valid)
models['xgb'] = xgb_model

preds_valid = xgb_model.predict_proba(X_valid)
preds_test = xgb_model.predict_proba(X_test)
predictions_valid['xgb'] = preds_valid
predictions_test['xgb'] = preds_test

metrics = ing_hubs_datathon_metric(y_valid, preds_valid)
model_metrics['xgb'] = metrics

print("="*60)
print("XGBoost (Hybrid) METR KLER")
print("="*60)
print(f"AUC:              {metrics['auc']:.5f}")
print(f"Gini:             {metrics['gini']:.5f}")
print(f"Recall@10%:       {metrics['recall@10']:.5f}")
print(f"Lift@10%:         {metrics['lift@10']:.5f}")
print("-"*60)
print(f"  Composite Score: {metrics['composite_score']:.5f}")
print("="*60)

gc.collect()

# 3. CatBoost
print("\n[3/4]   CatBoost (Hybrid) e itiliyor...")
cat_model = ModelWrapper('catboost', best_cat_params, config.SEED)
cat_model.fit(X_train, y_train, X_valid, y_valid, cat_features=cat_feature_names)
models['catboost'] = cat_model

preds_valid = cat_model.predict_proba(X_valid)
preds_test = cat_model.predict_proba(X_test)
predictions_valid['catboost'] = preds_valid
predictions_test['catboost'] = preds_test

metrics = ing_hubs_datathon_metric(y_valid, preds_valid)
model_metrics['catboost'] = metrics

print("="*60)
print("CatBoost (Hybrid) METR KLER")
print("="*60)
print(f"AUC:              {metrics['auc']:.5f}")
print(f"Gini:             {metrics['gini']:.5f}")
print(f"Recall@10%:       {metrics['recall@10']:.5f}")
print(f"Lift@10%:         {metrics['lift@10']:.5f}")
print("-"*60)
print(f"  Composite Score: {metrics['composite_score']:.5f}")
print("="*60)

gc.collect()

# 4. ExtraTrees
print("\n[4/4]   ExtraTrees (Hybrid) e itiliyor...")
et_params = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'bootstrap': False,
    'n_jobs': -1
}
et_model = ModelWrapper('extratrees', et_params, config.SEED)
et_model.fit(X_train, y_train)
models['extratrees'] = et_model

preds_valid = et_model.predict_proba(X_valid)
preds_test = et_model.predict_proba(X_test)
predictions_valid['extratrees'] = preds_valid
predictions_test['extratrees'] = preds_test

metrics = ing_hubs_datathon_metric(y_valid, preds_valid)
model_metrics['extratrees'] = metrics

print("="*60)
print("ExtraTrees (Hybrid) METR KLER")
print("="*60)
print(f"AUC:              {metrics['auc']:.5f}")
print(f"Gini:             {metrics['gini']:.5f}")
print(f"Recall@10%:       {metrics['recall@10']:.5f}")
print(f"Lift@10%:         {metrics['lift@10']:.5f}")
print("-"*60)
print(f"  Composite Score: {metrics['composite_score']:.5f}")
print("="*60)

gc.collect()

# In[ ]:
# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("  MODEL KAR ILA TIRMASI (Hybrid)")
print("="*80)

comparison_df = pd.DataFrame(model_metrics).T
comparison_df = comparison_df.sort_values('composite_score', ascending=False)
print(comparison_df.to_string())
print("="*80)


# In[ ]:
# ============================================================================
# ENSEMBLE
# ============================================================================
print("\n  SIMPLE ENSEMBLE OLU TURULUYOR...")

# Calculate weights based on composite scores
scores = np.array([model_metrics[m]['composite_score'] for m in models.keys()])
weights = scores / scores.sum()

# Ensure minimum weight of 15%
min_weight = 0.15
weights = np.maximum(weights, min_weight)
weights = weights / weights.sum()  # Normalize

print("\n  Model A  rl klar  (Normalized):")
for model_name, weight in zip(models.keys(), weights):
    print(f"   {model_name:15s}: {weight*100:.2f}%")

# Weighted average
ensemble_valid = sum(w * predictions_valid[m] for m, w in zip(models.keys(), weights))
ensemble_test = sum(w * predictions_test[m] for m, w in zip(models.keys(), weights))

# Evaluate ensemble
metrics_ensemble = ing_hubs_datathon_metric(y_valid, ensemble_valid)

print("\n" + "="*60)
print("ENSEMBLE (Raw - Hybrid) METR KLER")
print("="*60)
print(f"AUC:              {metrics_ensemble['auc']:.5f}")
print(f"Gini:             {metrics_ensemble['gini']:.5f}")
print(f"Recall@10%:       {metrics_ensemble['recall@10']:.5f}")
print(f"Lift@10%:         {metrics_ensemble['lift@10']:.5f}")
print("-"*60)
print(f"  Composite Score: {metrics_ensemble['composite_score']:.5f}")
print("="*60)

# In[ ]:
# ============================================================================
# CALIBRATION
# ============================================================================
print("\n  Isotonic Calibration uygulan yor...")

iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(ensemble_valid, y_valid)

ensemble_valid_calibrated = iso_reg.transform(ensemble_valid)
ensemble_test_calibrated = iso_reg.transform(ensemble_test)

metrics_calibrated = ing_hubs_datathon_metric(y_valid, ensemble_valid_calibrated)

print("="*60)
print("ENSEMBLE (Calibrated - Hybrid) METR KLER")
print("="*60)
print(f"AUC:              {metrics_calibrated['auc']:.5f}")
print(f"Gini:             {metrics_calibrated['gini']:.5f}")
print(f"Recall@10%:       {metrics_calibrated['recall@10']:.5f}")
print(f"Lift@10%:         {metrics_calibrated['lift@10']:.5f}")
print("-"*60)
print(f"  Composite Score: {metrics_calibrated['composite_score']:.5f}")
print("="*60)



# In[ ]:
# ============================================================================
# V1 vs HYBRID COMPARISON
# ============================================================================
print("\n" + "="*80)
print("                      V1 vs H BR T KAR ILA TIRMA")
print("="*80)

# V1 baseline scores (from prompt)
v1_validation_score = 1.17214
v1_kaggle_score = 1.17029
v1_gap = v1_validation_score - v1_kaggle_score

# Hybrid scores
hybrid_validation_score = metrics_calibrated['composite_score']
predicted_kaggle_score = hybrid_validation_score - v1_gap

# Calculate improvements
improvement_pct = ((hybrid_validation_score - v1_validation_score) / v1_validation_score) * 100
gain = hybrid_validation_score - v1_validation_score

print("\n  COMPOSITE SCORE KAR ILA TIRMASI:")
print("-"*80)
print(f"   V1 (Baseline - Validation):       {v1_validation_score:.5f}")
print(f"   V1 (Baseline - Kaggle):           {v1_kaggle_score:.5f}")
print(f"   Hybrid (Improved - Validation):   {hybrid_validation_score:.5f}")
print(f"    yile me:                          +{improvement_pct:.2f}%")
print(f"   Kazan :                            +{gain:.5f}")

print("\n  DETAYLI METR KLER (H BR T):")
print("-"*80)
print(f"   auc                 : {metrics_calibrated['auc']:.5f}")
print(f"   gini                : {metrics_calibrated['gini']:.5f}")
print(f"   recall@10           : {metrics_calibrated['recall@10']:.5f}")
print(f"   lift@10             : {metrics_calibrated['lift@10']:.5f}")
print(f"   composite_score     : {metrics_calibrated['composite_score']:.5f}")

print("\n  TAHM N  KAGGLE SKORU:")
print("-"*80)
print(f"   Hybrid Validation:                {hybrid_validation_score:.5f}")
print(f"   V1 Validation-Test Gap:           {v1_gap:.5f}")
print(f"   Tahmini Kaggle (Hybrid):          {predicted_kaggle_score:.5f}")
print(f"   V1 Kaggle:                        {v1_kaggle_score:.5f}")
print(f"   Beklenen Kazan :                  +{predicted_kaggle_score - v1_kaggle_score:.5f}")
print(f"   Beklenen  yile me:                +{improvement_pct:.2f}%")

print("\n    VAL DAT ON-TEST GAP KONTROL :")
print("-"*80)
if abs(v1_gap) < 0.01:
    print(f"     Gap  ok d   k ({v1_gap:.5f}) - Model iyi genelle iyor!")
elif abs(v1_gap) < 0.02:
    print(f"     Gap kabul edilebilir ({v1_gap:.5f}) - Model makul seviyede genelle iyor.")
else:
    print(f"     Gap y ksek ({v1_gap:.5f}) - Overfit riski var!")
print("="*80)


# In[ ]:
# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("  FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance from best model (LightGBM)
lgb_importance = models['lgbm'].get_feature_importance()
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_importance
}).sort_values('importance', ascending=False)

# Mark new features
new_features = ['prod_velocity_3m', 'prod_momentum_6m', 'prod_erosion_score',
                'engagement_decay', 'monetary_trend', 'lifetime_value_proxy',
                'age_tenure_interaction', 'prod_activity_6m', 'recency_product_risk']

feature_importance_df['is_new'] = feature_importance_df['feature'].isin(new_features)

print("\n  TOP 20  ZELL KLER (H BR T):")
print("="*70)
for idx, row in feature_importance_df.head(20).iterrows():
    marker = "  " if row['is_new'] else "    "
    print(f"{marker}{idx+1:2d}. {row['feature']:40s} ({row['importance']:,.2f})")
print("="*70)

# Check if new features are in top 50
new_in_top50 = feature_importance_df.head(50)['is_new'].sum()
print(f"\n  Yeni  zelliklerden top 50'de olan: {new_in_top50}/9")
if new_in_top50 >= 5:
    print("     Yeni  zellikler ba ar l !")
else:
    print("       Yeni  zellikler yeterince g  l  de il!")


# In[ ]:
# ============================================================================
# CREATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("  SUBMISSION OLU TURULUYOR...")
print("="*80)

submission = pd.DataFrame({
    'cust_id': test_features['cust_id'],
    'churn': ensemble_test_calibrated
})

# Sort by cust_id (critical for Kaggle)
submission = submission.sort_values('cust_id').reset_index(drop=True)

print("\n  SUBMISSION  STAT ST KLER :")
print(f"   Sat r say s :  {len(submission):,}")
print(f"   churn min:     {submission['churn'].min():.6f}")
print(f"   churn max:     {submission['churn'].max():.6f}")
print(f"   churn mean:    {submission['churn'].mean():.6f}")
print(f"   churn std:     {submission['churn'].std():.6f}")
print(f"   NaN say s :    {submission['churn'].isna().sum()}")

# Check if IDs are sorted
is_sorted = (submission['cust_id'].diff().dropna() > 0).all()
print(f"   {' ' if is_sorted else ' '} ID s ras  {'do ru' if is_sorted else 'YANLI '}!")

# Save submission
submission_path = f"{config.OUTPUT_DIR}submission_hybrid.csv"
submission.to_csv(submission_path, index=False)
print(f"\n  Submission kaydedildi: {submission_path}")
print(f"    lk 5 sat r:")
print(submission.head().to_string(index=False))
sys.exit(0)

# In[ ]:
# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("  G RSELLE T RMELER OLU TURULUYOR...")
print("="*80)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Model Comparison
ax = axes[0, 0]
comparison_plot = comparison_df.sort_values('composite_score')
ax.barh(comparison_plot.index, comparison_plot['composite_score'], color='steelblue')
ax.axvline(x=v1_validation_score, color='red', linestyle='--', linewidth=2, label=f'V1 Baseline ({v1_validation_score:.5f})')
ax.set_xlabel('Composite Score', fontsize=12)
ax.set_title('Model Comparison (Hybrid)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# 2. Ensemble Weights
ax = axes[0, 1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
wedges, texts, autotexts = ax.pie(
    weights,
    labels=list(models.keys()),
    autopct='%1.1f%%',
    startangle=90,
    colors=colors
)
ax.set_title('Ensemble Model Weights', fontsize=14, fontweight='bold')

# 3. Prediction Distribution
ax = axes[1, 0]
ax.hist(ensemble_test_calibrated, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(x=ensemble_test_calibrated.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ensemble_test_calibrated.mean():.4f}')
ax.set_xlabel('Predicted Churn Probability', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Test Set Prediction Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. V1 vs Hybrid Comparison
ax = axes[1, 1]
categories = ['V1\nValidation', 'V1\nKaggle', 'Hybrid\nValidation', 'Hybrid\n(Expected)']
scores = [v1_validation_score, v1_kaggle_score, hybrid_validation_score, predicted_kaggle_score]
colors_bar = ['#d62728', '#d62728', '#2ca02c', '#2ca02c']
bars = ax.bar(categories, scores, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel('Composite Score', fontsize=12)
ax.set_title('V1 vs Hybrid Performance', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.5f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_DIR}hybrid_model_analysis.png", dpi=300, bbox_inches='tight')
print("  G rselle tirme kaydedildi: hybrid_model_analysis.png")

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(12, 8))
top20 = feature_importance_df.head(20)
colors_feat = ['green' if is_new else 'steelblue' for is_new in top20['is_new']]
ax.barh(range(len(top20)), top20['importance'], color=colors_feat)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 20 Feature Importance (Green = New Features)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{config.OUTPUT_DIR}feature_importance.png", dpi=300, bbox_inches='tight')
print("  Feature importance plot kaydedildi: feature_importance.png")


# In[ ]:
# ============================================================================
# DOWNLOAD FILES (for Google Colab)
# ============================================================================
print("\n" + "="*80)
print("  DOSYALAR  ND R L YOR...")
print("="*80)

try:
    from google.colab import files

    # Create zip file
    import zipfile
    zip_path = f"{config.OUTPUT_DIR}hybrid_submission.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(f"{config.OUTPUT_DIR}submission_hybrid.csv", "submission_hybrid.csv")
        zipf.write(f"{config.OUTPUT_DIR}hybrid_model_analysis.png", "hybrid_model_analysis.png")
        zipf.write(f"{config.OUTPUT_DIR}feature_importance.png", "feature_importance.png")

    files.download(zip_path)
    print("  T m dosyalar indirildi: hybrid_submission.zip")

except ImportError:
    print("    Google Colab ortam  de il, dosyalar  u klas re kaydedildi:")
    print(f"   {config.OUTPUT_DIR}")

# In[ ]:
# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("  H BR T YAKLA IM TAMAMLANDI!")
print("="*80)

total_trials = config.LGB_TUNING_TRIALS + config.XGB_TUNING_TRIALS + config.CAT_TUNING_TRIALS
print("\n   SUMMARY:")
print(f"     Feature Count:       {len(feature_cols)} (V1: 157, New: 9)")
print(f"     Tuning Trials:       {total_trials} ({config.LGB_TUNING_TRIALS}+{config.XGB_TUNING_TRIALS}+{config.CAT_TUNING_TRIALS})")
print(f"     Models Used:         4")
print(f"     Validation Score:    {hybrid_validation_score:.5f}")
print(f"     Expected Kaggle:     {predicted_kaggle_score:.5f}")
print(f"     Expected Gain:       +{improvement_pct:.2f}%")

print("\n  SON KONTROL L STES :")
checks = [
    (len(feature_cols) == 166, f"Feature count = {len(feature_cols)} (beklenen: 166)"),
    (X_train.isna().sum().sum() == 0, "NaN count = 0"),
    (hybrid_validation_score > 1.19, f"Validation score > 1.19 ({hybrid_validation_score:.5f})"),
    (abs(v1_gap) < 0.01, f"Validation-test gap < 0.01 ({abs(v1_gap):.5f})"),
    (len(submission) == len(test_features), f"Submission sat r say s  = {len(submission):,}"),
    (is_sorted, "Submission ID s ras  do ru"),
    ((submission['churn'] >= 0).all() and (submission['churn'] <= 1).all(), "churn de erleri [0, 1] aral   nda"),
    (new_in_top50 >= 5, f"Yeni features top 50'de ({new_in_top50}/9)"),
]

for check, desc in checks:
    status = "OK" if check else "!!"
    print(f"   {status} {desc}")

print("\n  BA ARI DE ERLEND RMES :")
if hybrid_validation_score >= 1.19 and abs(v1_gap) < 0.02 and improvement_pct >= 2:
    print("     BA ARILI! Model hedeflere ula t .")
    print("     Submission'  Kaggle'a y kleyebilirsiniz!")
elif hybrid_validation_score >= 1.17 and improvement_pct >= 0:
    print("     ORTA BA ARI. Model  al   yor ama daha iyi olabilir.")
    print("     Feature engineering veya tuning iyile tirilebilir.")
else:
    print("     BA ARISIZ. Model beklenenden d   k performans g steriyor.")
    print("     Parametreleri g zden ge irin veya feature'lar  sadele tirin.")

print("\n  SONRAK  ADIM:")
print("   1. submission_hybrid.csv dosyas n  Kaggle'a y kle")
print("   2. Kaggle skorunu bu rapordaki tahminle kar  la t r")
print("   3. Gap > 0.02 ise overfitting i in feature engineering g zden ge ir")
print("\n" + "="*80)
print("  GOOD LUCK!")
print("="*80)

