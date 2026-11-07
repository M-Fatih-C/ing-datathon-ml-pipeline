#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ING Datathon - Veri Analizi ve Keşifsel Veri Analizi
Bu script veri setlerini detaylı analiz eder ve eksik değerleri tespit eder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Veri setlerini yükler ve temel analizleri yapar"""
    
    print("=" * 60)
    print("ING DATATHON - VERİ ANALİZİ")
    print("=" * 60)
    
    # Veri setlerini yükle
    print("\n1. VERİ SETLERİNİ YÜKLENİYOR...")
    
    try:
        customers = pd.read_csv('customers.csv')
        print(f"✓ customers.csv yüklendi: {customers.shape}")
        
        customer_history = pd.read_csv('customer_history.csv')
        print(f"✓ customer_history.csv yüklendi: {customer_history.shape}")
        
        reference_data = pd.read_csv('referance_data.csv')
        print(f"✓ referance_data.csv yüklendi: {reference_data.shape}")
        
        reference_test = pd.read_csv('referance_data_test.csv')
        print(f"✓ referance_data_test.csv yüklendi: {reference_test.shape}")
        
        sample_submission = pd.read_csv('sample_submission.csv')
        print(f"✓ sample_submission.csv yüklendi: {sample_submission.shape}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None
    
    # Temel bilgiler
    print("\n2. TEMEL VERİ BİLGİLERİ")
    print("-" * 40)
    
    datasets = {
        'customers': customers,
        'customer_history': customer_history,
        'reference_data': reference_data,
        'reference_test': reference_test,
        'sample_submission': sample_submission
    }
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Boyut: {df.shape}")
        print(f"  Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Sütunlar: {list(df.columns)}")
        
        # Eksik değerler
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"  Eksik değerler:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"  Eksik değer: Yok")
    
    # Churn analizi
    print("\n3. CHURN ANALİZİ")
    print("-" * 40)
    
    churn_stats = reference_data['churn'].value_counts()
    churn_rate = churn_stats[1] / len(reference_data) * 100
    
    print(f"Toplam müşteri sayısı: {len(reference_data):,}")
    print(f"Churn olan müşteri: {churn_stats[1]:,} ({churn_rate:.2f}%)")
    print(f"Churn olmayan müşteri: {churn_stats[0]:,} ({100-churn_rate:.2f}%)")
    
    # Tarih analizi
    print("\n4. TARİH ANALİZİ")
    print("-" * 40)
    
    reference_data['ref_date'] = pd.to_datetime(reference_data['ref_date'])
    reference_test['ref_date'] = pd.to_datetime(reference_test['ref_date'])
    
    print(f"Train referans tarihleri:")
    print(f"  En erken: {reference_data['ref_date'].min()}")
    print(f"  En geç: {reference_data['ref_date'].max()}")
    
    print(f"Test referans tarihleri:")
    print(f"  En erken: {reference_test['ref_date'].min()}")
    print(f"  En geç: {reference_test['ref_date'].max()}")
    
    # Müşteri geçmişi analizi
    print("\n5. MÜŞTERİ GEÇMİŞİ ANALİZİ")
    print("-" * 40)
    
    customer_history['date'] = pd.to_datetime(customer_history['date'])
    
    print(f"İşlem tarihleri:")
    print(f"  En erken: {customer_history['date'].min()}")
    print(f"  En geç: {customer_history['date'].max()}")
    
    # Her müşterinin ortalama işlem sayısı
    transactions_per_customer = customer_history.groupby('cust_id').size()
    print(f"Müşteri başına ortalama işlem sayısı: {transactions_per_customer.mean():.2f}")
    print(f"Müşteri başına min işlem sayısı: {transactions_per_customer.min()}")
    print(f"Müşteri başına max işlem sayısı: {transactions_per_customer.max()}")
    
    # Demografik analiz
    print("\n6. DEMOGRAFİK ANALİZ")
    print("-" * 40)
    
    print("Cinsiyet dağılımı:")
    print(customers['gender'].value_counts())
    
    print("\nYaş istatistikleri:")
    print(f"  Ortalama yaş: {customers['age'].mean():.2f}")
    print(f"  Medyan yaş: {customers['age'].median():.2f}")
    print(f"  Min yaş: {customers['age'].min()}")
    print(f"  Max yaş: {customers['age'].max()}")
    
    print("\nBölge dağılımı (Top 10):")
    print(customers['province'].value_counts().head(10))
    
    print("\nÇalışma durumu:")
    print(customers['work_type'].value_counts())
    
    return datasets

def analyze_missing_values(datasets):
    """Eksik değerleri detaylı analiz eder"""
    
    print("\n7. EKSİK DEĞER DETAY ANALİZİ")
    print("-" * 40)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        
        # Eksik değer matrisi
        missing_matrix = df.isnull()
        missing_count = missing_matrix.sum()
        
        if missing_count.sum() > 0:
            print("Eksik değer dağılımı:")
            for col in missing_count[missing_count > 0].index:
                missing_pct = missing_count[col] / len(df) * 100
                print(f"  {col}: {missing_count[col]:,} ({missing_pct:.2f}%)")
                
                # Eksik değerlerin diğer sütunlarla ilişkisi
                if len(df.columns) > 1:
                    corr_missing = missing_matrix[col].corr(missing_matrix.drop(col, axis=1).sum(axis=1))
                    print(f"    Diğer eksik değerlerle korelasyon: {corr_missing:.3f}")
        else:
            print("  Eksik değer yok")

def create_summary_report(datasets):
    """Özet rapor oluşturur"""
    
    print("\n8. ÖZET RAPOR")
    print("=" * 60)
    
    # Veri kalitesi skoru
    total_rows = sum(len(df) for df in datasets.values())
    total_missing = sum(df.isnull().sum().sum() for df in datasets.values())
    data_quality_score = (1 - total_missing / (total_rows * len(datasets))) * 100
    
    print(f"Toplam veri satırı: {total_rows:,}")
    print(f"Toplam eksik değer: {total_missing:,}")
    print(f"Veri kalitesi skoru: {data_quality_score:.2f}%")
    
    # Öneriler
    print("\nÖNERİLER:")
    print("1. Feature engineering için temporal analiz yapılmalı")
    print("2. Müşteri segmentasyonu için clustering uygulanmalı")
    print("3. Eksik değerler için uygun imputation stratejisi belirlenmeli")
    print("4. Time-aware validation stratejisi kullanılmalı")
    
    return datasets

if __name__ == "__main__":
    # Ana analiz
    datasets = load_and_analyze_data()
    
    if datasets:
        # Eksik değer analizi
        analyze_missing_values(datasets)
        
        # Özet rapor
        create_summary_report(datasets)
        
        print("\n" + "=" * 60)
        print("VERİ ANALİZİ TAMAMLANDI!")
        print("=" * 60)
