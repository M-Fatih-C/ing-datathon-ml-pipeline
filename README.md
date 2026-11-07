# ING Datathon ML Pipeline

ING Türkiye Datathon çalışması için hazırlanmış uçtan uca müşteri davranışı
modelleme ve olasılık tahmini hattı.

## İçerik

- Veri keşfi ve kalite kontrolleri
- Tarihsel müşteri hareketlerinden özellik üretimi
- Bellek kullanımına göre optimize edilmiş özellik mühendisliği
- LightGBM, XGBoost, CatBoost ve klasik modeller
- Ensemble/blending yaklaşımı
- Optuna ile hiperparametre optimizasyonu
- Notebook ve çalıştırılabilir Python sürümü

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Kodlar yarışma dosyalarını çalışma dizininde bekler. Veri setleri ve submission
çıktıları lisans, boyut ve gizlilik nedenleriyle bu depoya eklenmemiştir.

## Proje durumu

Bu depo bir yarışma/araştırma arşividir. Kod, farklı modelleme yaklaşımlarını ve
deney akışlarını korumak amacıyla yayımlanmıştır; üretim servisi değildir.

