#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ING Datathon - Kapsamlı Feature Engineering
Bu script müşteri verilerinden en güçlü özellikleri çıkarır.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Kapsamlı feature engineering sınıfı"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_temporal_features(self, df_history, ref_date):
        """Temporal (zaman bazlı) özellikler oluşturur"""
        
        print("Temporal features oluşturuluyor...")
        
        # Referans tarihinden önceki verileri filtrele
        df_history['date'] = pd.to_datetime(df_history['date'])
        ref_date = pd.to_datetime(ref_date)
        
        # Son 6, 12, 18, 24 ay için veriler
        periods = [6, 12, 18, 24]
        features = {}
        
        for period in periods:
            cutoff_date = ref_date - pd.DateOffset(months=period)
            period_data = df_history[df_history['date'] >= cutoff_date].copy()
            
            if len(period_data) == 0:
                continue
                
            # Her müşteri için özellikler
            customer_features = period_data.groupby('cust_id').agg({
                # Transaction counts
                'mobile_eft_all_cnt': ['sum', 'mean', 'std', 'max', 'min'],
                'cc_transaction_all_cnt': ['sum', 'mean', 'std', 'max', 'min'],
                'active_product_category_nbr': ['sum', 'mean', 'std', 'max', 'min'],
                
                # Transaction amounts
                'mobile_eft_all_amt': ['sum', 'mean', 'std', 'max', 'min'],
                'cc_transaction_all_amt': ['sum', 'mean', 'std', 'max', 'min'],
                
                # Date features
                'date': ['count', 'min', 'max']
            }).fillna(0)
            
            # Sütun isimlerini düzelt
            customer_features.columns = [f'{period}m_{col[0]}_{col[1]}' for col in customer_features.columns]
            
            # Trend features
            customer_features[f'{period}m_trend_mobile_eft'] = (
                period_data.groupby('cust_id')['mobile_eft_all_amt']
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            )
            
            customer_features[f'{period}m_trend_cc'] = (
                period_data.groupby('cust_id')['cc_transaction_all_amt']
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            )
            
            # Volatility features
            customer_features[f'{period}m_volatility_mobile_eft'] = (
                period_data.groupby('cust_id')['mobile_eft_all_amt'].std()
            )
            customer_features[f'{period}m_volatility_cc'] = (
                period_data.groupby('cust_id')['cc_transaction_all_amt'].std()
            )
            
            # Activity patterns
            customer_features[f'{period}m_activity_months'] = (
                period_data.groupby('cust_id')['date'].nunique()
            )
            
            # Spending concentration
            customer_features[f'{period}m_mobile_eft_concentration'] = (
                customer_features[f'{period}m_mobile_eft_all_amt_max'] / 
                (customer_features[f'{period}m_mobile_eft_all_amt_sum'] + 1e-8)
            )
            
            customer_features[f'{period}m_cc_concentration'] = (
                customer_features[f'{period}m_cc_transaction_all_amt_max'] / 
                (customer_features[f'{period}m_cc_transaction_all_amt_sum'] + 1e-8)
            )
            
            features.update(customer_features.to_dict('index'))
        
        return pd.DataFrame.from_dict(features, orient='index').fillna(0)
    
    def create_behavioral_features(self, df_history, ref_date):
        """Davranışsal özellikler oluşturur"""
        
        print("Behavioral features oluşturuluyor...")
        
        df_history['date'] = pd.to_datetime(df_history['date'])
        ref_date = pd.to_datetime(ref_date)
        
        # Son 12 ay verisi
        cutoff_date = ref_date - pd.DateOffset(months=12)
        recent_data = df_history[df_history['date'] >= cutoff_date].copy()
        
        if len(recent_data) == 0:
            return pd.DataFrame()
        
        # Müşteri bazında davranışsal özellikler
        behavioral_features = recent_data.groupby('cust_id').agg({
            # Transaction frequency patterns
            'mobile_eft_all_cnt': ['sum', 'mean', 'std'],
            'cc_transaction_all_cnt': ['sum', 'mean', 'std'],
            
            # Spending patterns
            'mobile_eft_all_amt': ['sum', 'mean', 'std', 'skew'],
            'cc_transaction_all_amt': ['sum', 'mean', 'std', 'skew'],
            
            # Product usage
            'active_product_category_nbr': ['sum', 'mean', 'std', 'max']
        }).fillna(0)
        
        behavioral_features.columns = [f'behavior_{col[0]}_{col[1]}' for col in behavioral_features.columns]
        
        # Payment method preferences
        behavioral_features['payment_preference_mobile'] = (
            behavioral_features['behavior_mobile_eft_all_amt_sum'] / 
            (behavioral_features['behavior_mobile_eft_all_amt_sum'] + 
             behavioral_features['behavior_cc_transaction_all_amt_sum'] + 1e-8)
        )
        
        # Transaction size patterns
        behavioral_features['avg_transaction_size_mobile'] = (
            behavioral_features['behavior_mobile_eft_all_amt_mean'] / 
            (behavioral_features['behavior_mobile_eft_all_cnt_mean'] + 1e-8)
        )
        
        behavioral_features['avg_transaction_size_cc'] = (
            behavioral_features['behavior_cc_transaction_all_amt_mean'] / 
            (behavioral_features['behavior_cc_transaction_all_cnt_mean'] + 1e-8)
        )
        
        # Activity consistency
        behavioral_features['activity_consistency'] = (
            1 - behavioral_features['behavior_mobile_eft_all_cnt_std'] / 
            (behavioral_features['behavior_mobile_eft_all_cnt_mean'] + 1e-8)
        )
        
        # Spending growth rate
        monthly_spending = recent_data.groupby(['cust_id', recent_data['date'].dt.to_period('M')])[
            ['mobile_eft_all_amt', 'cc_transaction_all_amt']
        ].sum().groupby('cust_id')
        
        spending_growth = []
        for cust_id, group in monthly_spending:
            total_spending = group['mobile_eft_all_amt'] + group['cc_transaction_all_amt']
            if len(total_spending) > 1:
                growth_rate = np.polyfit(range(len(total_spending)), total_spending, 1)[0]
            else:
                growth_rate = 0
            spending_growth.append((cust_id, growth_rate))
        
        growth_df = pd.DataFrame(spending_growth, columns=['cust_id', 'spending_growth_rate'])
        growth_df.set_index('cust_id', inplace=True)
        
        behavioral_features = behavioral_features.join(growth_df, how='left')
        behavioral_features['spending_growth_rate'] = behavioral_features['spending_growth_rate'].fillna(0)
        
        return behavioral_features
    
    def create_demographic_features(self, df_customers):
        """Demografik özellikler oluşturur"""
        
        print("Demographic features oluşturuluyor...")
        
        features = df_customers.copy()
        
        # Yaş grupları
        features['age_group'] = pd.cut(features['age'], 
                                     bins=[0, 25, 35, 45, 55, 65, 100], 
                                     labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        # Tenure grupları
        features['tenure_group'] = pd.cut(features['tenure'], 
                                        bins=[0, 12, 24, 36, 60, 120, 300], 
                                        labels=['0-1y', '1-2y', '2-3y', '3-5y', '5-10y', '10y+'])
        
        # Çalışma durumu kombinasyonları
        features['work_status'] = features['work_type'].map({
            'Full-time': 'Active',
            'Part-time': 'Active', 
            'Self-employed': 'Active',
            'Unemployed': 'Inactive',
            'Student': 'Inactive',
            'Retired': 'Inactive'
        })
        
        # Risk skorları (yaş ve tenure bazlı)
        features['age_risk_score'] = np.where(features['age'] < 30, 1.2,
                                           np.where(features['age'] > 60, 1.1, 1.0))
        
        features['tenure_risk_score'] = np.where(features['tenure'] < 12, 1.3,
                                               np.where(features['tenure'] > 60, 0.9, 1.0))
        
        # Bölge risk skorları (churn oranına göre)
        province_risk = {
            'ZUI': 1.0, 'NOH': 1.1, 'NOB': 1.0, 'GEL': 0.9, 'UTR': 1.0,
            'OVE': 1.1, 'LIM': 1.0, 'FRI': 1.2, 'GRO': 1.1, 'DRE': 1.0
        }
        features['province_risk_score'] = features['province'].map(province_risk).fillna(1.0)
        
        # Çalışma sektörü risk skorları
        sector_risk = {
            'Technology': 0.9, 'Finance': 0.8, 'Healthcare': 0.9, 'Retail': 1.1,
            'Public Sector': 0.8, 'Education': 0.9, 'Manufacturing': 1.0,
            'Construction': 1.2, 'Transportation': 1.1, 'Other': 1.0
        }
        features['sector_risk_score'] = features['work_sector'].map(sector_risk).fillna(1.0)
        
        # Kombine risk skoru
        features['combined_risk_score'] = (
            features['age_risk_score'] * 
            features['tenure_risk_score'] * 
            features['province_risk_score'] * 
            features['sector_risk_score']
        )
        
        return features
    
    def create_interaction_features(self, df_temporal, df_behavioral, df_demographic):
        """Özellikler arası etkileşim özellikleri oluşturur"""
        
        print("Interaction features oluşturuluyor...")
        
        # Tüm özellikleri birleştir
        df_combined = df_demographic.copy()
        
        if not df_temporal.empty:
            df_combined = df_combined.join(df_temporal, how='left')
        
        if not df_behavioral.empty:
            df_combined = df_combined.join(df_behavioral, how='left')
        
        # Eksik değerleri doldur
        df_combined = df_combined.fillna(0)
        
        # Yaş ve harcama etkileşimi
        if '12m_mobile_eft_all_amt_sum' in df_combined.columns:
            df_combined['age_spending_interaction'] = (
                df_combined['age'] * df_combined['12m_mobile_eft_all_amt_sum']
            )
        
        # Tenure ve aktivite etkileşimi
        if '12m_mobile_eft_all_cnt_sum' in df_combined.columns:
            df_combined['tenure_activity_interaction'] = (
                df_combined['tenure'] * df_combined['12m_mobile_eft_all_cnt_sum']
            )
        
        # Risk skoru ve davranış etkileşimi
        if 'behavior_mobile_eft_all_amt_sum' in df_combined.columns:
            df_combined['risk_behavior_interaction'] = (
                df_combined['combined_risk_score'] * df_combined['behavior_mobile_eft_all_amt_sum']
            )
        
        return df_combined
    
    def create_advanced_features(self, df_history, ref_date):
        """Gelişmiş özellikler oluşturur"""
        
        print("Advanced features oluşturuluyor...")
        
        df_history['date'] = pd.to_datetime(df_history['date'])
        ref_date = pd.to_datetime(ref_date)
        
        # Son 6 ay verisi
        cutoff_date = ref_date - pd.DateOffset(months=6)
        recent_data = df_history[df_history['date'] >= cutoff_date].copy()
        
        if len(recent_data) == 0:
            return pd.DataFrame()
        
        advanced_features = {}
        
        for cust_id in recent_data['cust_id'].unique():
            cust_data = recent_data[recent_data['cust_id'] == cust_id].sort_values('date')
            
            if len(cust_data) < 2:
                continue
            
            # Transaction frequency decay
            dates = cust_data['date'].values
            time_diff = np.diff(dates) / np.timedelta64(1, 'D')
            avg_frequency = np.mean(time_diff) if len(time_diff) > 0 else 0
            
            # Spending momentum
            spending = cust_data['mobile_eft_all_amt'] + cust_data['cc_transaction_all_amt']
            if len(spending) > 1:
                momentum = np.polyfit(range(len(spending)), spending, 1)[0]
            else:
                momentum = 0
            
            # Activity burst detection
            activity_counts = cust_data['mobile_eft_all_cnt'] + cust_data['cc_transaction_all_cnt']
            burst_threshold = activity_counts.mean() + 2 * activity_counts.std()
            burst_count = (activity_counts > burst_threshold).sum()
            
            # Seasonal patterns
            cust_data['month'] = cust_data['date'].dt.month
            monthly_spending = cust_data.groupby('month')['mobile_eft_all_amt'].sum()
            seasonal_variance = monthly_spending.var() if len(monthly_spending) > 1 else 0
            
            advanced_features[cust_id] = {
                'transaction_frequency_decay': avg_frequency,
                'spending_momentum': momentum,
                'activity_burst_count': burst_count,
                'seasonal_variance': seasonal_variance,
                'transaction_consistency': 1 - (activity_counts.std() / (activity_counts.mean() + 1e-8))
            }
        
        return pd.DataFrame.from_dict(advanced_features, orient='index').fillna(0)
    
    def engineer_all_features(self, df_customers, df_history, df_reference):
        """Tüm özellikleri oluşturur"""
        
        print("=" * 60)
        print("KAPSAMLI FEATURE ENGINEERING BAŞLIYOR")
        print("=" * 60)
        
        all_features = []
        
        for idx, row in df_reference.iterrows():
            cust_id = row['cust_id']
            ref_date = row['ref_date']
            
            if idx % 10000 == 0:
                print(f"İşlenen müşteri: {idx}/{len(df_reference)}")
            
            # Müşteri verilerini filtrele
            cust_customers = df_customers[df_customers['cust_id'] == cust_id]
            cust_history = df_history[df_history['cust_id'] == cust_id]
            
            if len(cust_customers) == 0 or len(cust_history) == 0:
                continue
            
            # Özellikleri oluştur
            temporal_features = self.create_temporal_features(cust_history, ref_date)
            behavioral_features = self.create_behavioral_features(cust_history, ref_date)
            demographic_features = self.create_demographic_features(cust_customers)
            advanced_features = self.create_advanced_features(cust_history, ref_date)
            
            # Özellikleri birleştir
            combined_features = self.create_interaction_features(
                temporal_features, behavioral_features, demographic_features
            )
            
            if not advanced_features.empty:
                combined_features = combined_features.join(advanced_features, how='left')
            
            # Churn etiketini ekle
            combined_features['churn'] = row['churn']
            combined_features['cust_id'] = cust_id
            combined_features['ref_date'] = ref_date
            
            all_features.append(combined_features)
        
        # Tüm özellikleri birleştir
        final_features = pd.concat(all_features, ignore_index=True)
        
        print(f"\nToplam özellik sayısı: {len(final_features.columns)}")
        print(f"Toplam müşteri sayısı: {len(final_features)}")
        
        return final_features

def main():
    """Ana fonksiyon"""
    
    # Veri setlerini yükle
    print("Veri setleri yükleniyor...")
    customers = pd.read_csv('customers.csv')
    customer_history = pd.read_csv('customer_history.csv')
    reference_data = pd.read_csv('referance_data.csv')
    
    # Feature engineer oluştur
    fe = FeatureEngineer()
    
    # Özellikleri oluştur
    features = fe.engineer_all_features(customers, customer_history, reference_data)
    
    # Özellikleri kaydet
    features.to_csv('engineered_features.csv', index=False)
    print("\nÖzellikler 'engineered_features.csv' dosyasına kaydedildi!")
    
    return features

if __name__ == "__main__":
    features = main()
