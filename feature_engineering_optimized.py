#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ING Datathon - Optimized Feature Engineering
Hızlı ve etkili özellik çıkarımı
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_features_optimized():
    """Optimize edilmiş feature engineering"""
    
    print("=" * 60)
    print("OPTIMIZE EDİLMİŞ FEATURE ENGINEERING")
    print("=" * 60)
    
    # Veri setlerini yükle
    print("\n1. Veri setleri yükleniyor...")
    customers = pd.read_csv('customers.csv')
    customer_history = pd.read_csv('customer_history.csv')
    reference_data = pd.read_csv('referance_data.csv')
    reference_test = pd.read_csv('referance_data_test.csv')
    
    print(f"[OK] customers: {customers.shape}")
    print(f"[OK] customer_history: {customer_history.shape}")
    print(f"[OK] reference_data: {reference_data.shape}")
    print(f"[OK] reference_test: {reference_test.shape}")
    
    # Tarih sütunlarını dönüştür
    customer_history['date'] = pd.to_datetime(customer_history['date'])
    reference_data['ref_date'] = pd.to_datetime(reference_data['ref_date'])
    reference_test['ref_date'] = pd.to_datetime(reference_test['ref_date'])
    
    # Eksik değerleri doldur
    print("\n2. Eksik değerler dolduruluyor...")
    customer_history = customer_history.fillna(0)
    customers['work_sector'] = customers['work_sector'].fillna('Unknown')
    
    # Train ve test verilerini birleştir
    print("\n3. Train ve test verileri birleştiriliyor...")
    reference_data['is_train'] = True
    reference_test['is_train'] = False
    reference_test['churn'] = 0  # Placeholder
    
    all_reference = pd.concat([reference_data, reference_test], ignore_index=True)
    
    # Her müşteri için özellikler oluştur
    print("\n4. Özellikler oluşturuluyor...")
    
    features_list = []
    
    for idx, row in all_reference.iterrows():
        if idx % 5000 == 0:
            print(f"İşlenen müşteri: {idx}/{len(all_reference)}")
        
        cust_id = row['cust_id']
        ref_date = row['ref_date']
        
        # Müşteri bilgilerini al
        cust_info = customers[customers['cust_id'] == cust_id]
        if len(cust_info) == 0:
            continue
        
        cust_info = cust_info.iloc[0]
        
        # Müşteri geçmişini al
        cust_history = customer_history[customer_history['cust_id'] == cust_id].copy()
        if len(cust_history) == 0:
            continue
        
        # Referans tarihinden önceki verileri filtrele
        cust_history = cust_history[cust_history['date'] < ref_date]
        
        if len(cust_history) == 0:
            continue
        
        # Temporal features (son 6, 12, 24 ay)
        features = {}
        
        for period in [6, 12, 24]:
            cutoff_date = ref_date - pd.DateOffset(months=period)
            period_data = cust_history[cust_history['date'] >= cutoff_date]
            
            if len(period_data) == 0:
                continue
            
            # Transaction counts
            features[f'{period}m_mobile_eft_cnt_sum'] = period_data['mobile_eft_all_cnt'].sum()
            features[f'{period}m_mobile_eft_cnt_mean'] = period_data['mobile_eft_all_cnt'].mean()
            features[f'{period}m_mobile_eft_cnt_std'] = period_data['mobile_eft_all_cnt'].std()
            
            features[f'{period}m_cc_cnt_sum'] = period_data['cc_transaction_all_cnt'].sum()
            features[f'{period}m_cc_cnt_mean'] = period_data['cc_transaction_all_cnt'].mean()
            features[f'{period}m_cc_cnt_std'] = period_data['cc_transaction_all_cnt'].std()
            
            # Transaction amounts
            features[f'{period}m_mobile_eft_amt_sum'] = period_data['mobile_eft_all_amt'].sum()
            features[f'{period}m_mobile_eft_amt_mean'] = period_data['mobile_eft_all_amt'].mean()
            features[f'{period}m_mobile_eft_amt_std'] = period_data['mobile_eft_all_amt'].std()
            
            features[f'{period}m_cc_amt_sum'] = period_data['cc_transaction_all_amt'].sum()
            features[f'{period}m_cc_amt_mean'] = period_data['cc_transaction_all_amt'].mean()
            features[f'{period}m_cc_amt_std'] = period_data['cc_transaction_all_amt'].std()
            
            # Product categories
            features[f'{period}m_product_cat_sum'] = period_data['active_product_category_nbr'].sum()
            features[f'{period}m_product_cat_mean'] = period_data['active_product_category_nbr'].mean()
            features[f'{period}m_product_cat_max'] = period_data['active_product_category_nbr'].max()
            
            # Activity months
            features[f'{period}m_active_months'] = period_data['date'].nunique()
            
            # Spending ratios
            total_spending = features[f'{period}m_mobile_eft_amt_sum'] + features[f'{period}m_cc_amt_sum']
            if total_spending > 0:
                features[f'{period}m_mobile_eft_ratio'] = features[f'{period}m_mobile_eft_amt_sum'] / total_spending
                features[f'{period}m_cc_ratio'] = features[f'{period}m_cc_amt_sum'] / total_spending
            else:
                features[f'{period}m_mobile_eft_ratio'] = 0
                features[f'{period}m_cc_ratio'] = 0
        
        # Behavioral features
        if len(cust_history) > 1:
            # Trend analysis
            cust_history_sorted = cust_history.sort_values('date')
            
            # Mobile EFT trend
            mobile_eft_trend = np.polyfit(range(len(cust_history_sorted)), 
                                        cust_history_sorted['mobile_eft_all_amt'], 1)[0]
            features['mobile_eft_trend'] = mobile_eft_trend
            
            # CC trend
            cc_trend = np.polyfit(range(len(cust_history_sorted)), 
                                cust_history_sorted['cc_transaction_all_amt'], 1)[0]
            features['cc_trend'] = cc_trend
            
            # Activity trend
            activity_trend = np.polyfit(range(len(cust_history_sorted)), 
                                      cust_history_sorted['mobile_eft_all_cnt'] + 
                                      cust_history_sorted['cc_transaction_all_cnt'], 1)[0]
            features['activity_trend'] = activity_trend
        
        # Demographic features
        features['age'] = cust_info['age']
        features['tenure'] = cust_info['tenure']
        features['gender'] = cust_info['gender']
        features['province'] = cust_info['province']
        features['work_type'] = cust_info['work_type']
        features['work_sector'] = cust_info['work_sector']
        
        # Age groups
        if cust_info['age'] < 30:
            features['age_group'] = 'young'
        elif cust_info['age'] < 50:
            features['age_group'] = 'middle'
        else:
            features['age_group'] = 'senior'
        
        # Tenure groups
        if cust_info['tenure'] < 12:
            features['tenure_group'] = 'new'
        elif cust_info['tenure'] < 36:
            features['tenure_group'] = 'medium'
        else:
            features['tenure_group'] = 'loyal'
        
        # Risk scores
        features['age_risk'] = 1.2 if cust_info['age'] < 30 else (1.1 if cust_info['age'] > 60 else 1.0)
        features['tenure_risk'] = 1.3 if cust_info['tenure'] < 12 else (0.9 if cust_info['tenure'] > 60 else 1.0)
        
        # Interaction features
        features['age_tenure_interaction'] = cust_info['age'] * cust_info['tenure']
        
        if '12m_mobile_eft_amt_sum' in features:
            features['age_spending_interaction'] = cust_info['age'] * features['12m_mobile_eft_amt_sum']
        
        # Target variable
        features['churn'] = row['churn']
        features['is_train'] = row['is_train']
        features['cust_id'] = cust_id
        features['ref_date'] = ref_date
        
        features_list.append(features)
    
    # DataFrame oluştur
    print("\n5. DataFrame oluşturuluyor...")
    features_df = pd.DataFrame(features_list)
    
    # Eksik değerleri doldur
    features_df = features_df.fillna(0)
    
    print(f"\nToplam özellik sayısı: {len(features_df.columns)}")
    print(f"Toplam müşteri sayısı: {len(features_df)}")
    
    # Train ve test ayrımı
    train_features = features_df[features_df['is_train'] == True].copy()
    test_features = features_df[features_df['is_train'] == False].copy()
    
    print(f"Train müşteri sayısı: {len(train_features)}")
    print(f"Test müşteri sayısı: {len(test_features)}")
    
    # Dosyaları kaydet
    train_features.to_csv('train_features.csv', index=False)
    test_features.to_csv('test_features.csv', index=False)
    
    print("\n[OK] Train özellikleri 'train_features.csv' dosyasına kaydedildi!")
    print("[OK] Test özellikleri 'test_features.csv' dosyasına kaydedildi!")
    
    return train_features, test_features

if __name__ == "__main__":
    train_features, test_features = create_features_optimized()
