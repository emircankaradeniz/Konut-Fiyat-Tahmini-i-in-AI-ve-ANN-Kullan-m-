import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

"""
Gerekli kütüphaneler içe aktarılıyor:
pandas: Veri manipülasyonu ve analiz için.
numpy: Sayısal hesaplamalar için.
sklearn: Veri işleme, modelleme ve değerlendirme araçları.
matplotlib: Grafik oluşturmak için.
"""

file_path = 'data/tcmb_aylik_veriler.csv'
data = pd.read_csv(file_path)

"""
Veri seti CSV dosyasından yükleniyor ve `data` adlı bir pandas DataFrame'e aktarılıyor.
`file_path`: CSV dosyasının yolu.
`pd.read_csv`: CSV dosyasını okur ve bir DataFrame oluşturur.
"""

data['Tarih'] = pd.to_datetime(data['Tarih'], errors='coerce')
data['Yil'] = data['Tarih'].dt.year

"""
"Tarih" sütunu datetime formatına dönüştürülüyor:
`.dt.year`: Tarih sütunundan yıl bilgisi çıkarılıyor ve "Yil" adlı yeni bir sütuna ekleniyor.
`errors='coerce'`: Hatalı tarih değerlerini NaT (Not a Time) olarak işaretler.
"""

"""Eksik verileri medyan ile doldur"""
data = data.dropna(subset=['konut_fiyat_endeksi'])
data = data.fillna(data.median())


"""Eğitim ve test setlerini oluştur"""
train_data = data[data['Yil'].between(2020, 2023)]
test_data = data[data['Yil'] == 2024]

"""
Eğitim ve test veri setleri oluşturuluyor:
2020-2023 yılları arasındaki veriler eğitim seti olarak ayrılıyor (`train_data`).
2024 yılı verileri test seti olarak ayrılıyor (`test_data`).
"""

features = data.columns.difference(['Tarih', 'konut_fiyat_endeksi', 'Yil'])
X_train = train_data[features]
y_train = train_data['konut_fiyat_endeksi']
X_test = test_data[features]
y_test = test_data['konut_fiyat_endeksi']

"""
Giriş (X) ve hedef (y) değişkenleri ayrıştırılıyor:
`features`: Hedef değişken ve tarih sütunları dışındaki tüm sütunlar giriş değişkenleri olarak seçiliyor.
`X_train`: Eğitim setindeki giriş değişkenleri.
`y_train`: Eğitim setindeki hedef değişken (konut_fiyat_endeksi).
`X_test` ve `y_test`: Test setindeki giriş ve hedef değişkenler.
"""

"""Giriş ve hedef değişkenleri ölçeklendirelim"""
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

"""
Giriş ve hedef değişkenleri ölçeklendirme:
`StandardScaler`: Veriyi standartlaştırır (ortalama=0, standart sapma=1).
`fit_transform`: Eğitim verisini ölçeklendirir ve ölçekleme modelini öğrenir.
`transform`: Test verisini aynı ölçekleme modeliyle dönüştürür.
`reshape(-1, 1)`: Hedef değişken 2D formata çevrilir.
`ravel()`: 1D diziye dönüştürülür.
"""

"""Modeli oluşturma ve eğitme"""
model_relu = MLPRegressor(
    hidden_layer_sizes=(64,),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42
)

model_tanh = MLPRegressor(
    hidden_layer_sizes=(32,),
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42
)

"""SGD (Stochastic Gradient Descent) optimizasyon yöntemi ile ağırlıklar güncellenir."""

"""
İki model tanımlanıyor:
İlk model (`model_relu`): ReLU aktivasyon fonksiyonu ile 64 nöronlu tek katman.
İkinci model (`model_tanh`): Tanh aktivasyon fonksiyonu ile 32 nöronlu tek katman.
`solver='sgd'`: Stokastik gradyan inişi optimizasyon yöntemi.
`learning_rate_init=0.01`: Öğrenme oranı.
`max_iter=500`: Maksimum epoch sayısı.
`random_state=42`: Rastgelelik için sabit bir seed.
"""

"""Relu modeli eğit"""
model_relu.fit(X_train, y_train)
X_relu_output = model_relu.predict(X_train)

"""
İlk model eğitiliyor ve eğitim verisi üzerinden tahmin yapılıyor:
`model_relu.fit`: İlk modeli, eğitim seti üzerinde eğitir.
`model_relu.predict`: İlk modelin çıktısı, ikinci modelin girdi verisi olarak kullanılmak üzere saklanır.
"""

""" tanh modeli eğit"""
model_tanh.fit(X_relu_output.reshape(-1, 1), y_train)

"""
İkinci model eğitiliyor:
İlk modelin çıktısı (`X_relu_output`), ikinci modelin giriş verisi olarak kullanılıyor.
"""

"""Test setinde tahmin yap"""
X_test_relu_output = model_relu.predict(X_test)
y_pred = model_tanh.predict(X_test_relu_output.reshape(-1, 1))

"""
Test seti üzerinde tahmin yapılıyor:
İlk model, test setindeki giriş verileri üzerinde tahmin yapar.
İkinci model, bu tahminleri kullanarak son tahminleri üretir.
"""

"""Tahminleri geri dönüştür"""
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

"""
Tahminler ve gerçek değerler, orijinal ölçeğe geri dönüştürülüyor:
inverse_transform`: Ölçeklendirilmiş veriyi orijinal formata çevirir.
"""

"""Performans ölçümleri"""
mse = mean_squared_error(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')

"""
Model performansı değerlendiriliyor:
MSE (Mean Squared Error): Ortalama karesel hata.
MAE (Mean Absolute Error): Ortalama mutlak hata.
"""

plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Gerçek Değerler', marker='o')
plt.plot(y_pred, label='Tahminler', marker='x')
plt.title('Konut Fiyat Endeksi Tahmini (2024)')
plt.xlabel('Ay')
plt.ylabel('Konut Fiyat Endeksi')
plt.legend()
plt.grid()
plt.show()

""" Eğitim kayıp eğrisi"""
plt.figure(figsize=(10, 6))
plt.plot(model_relu.loss_curve_, label='ReLU Eğitim Kaybı')
plt.plot(model_tanh.loss_curve_, label='Tanh Eğitim Kaybı')
plt.title('Eğitim Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

for i in range(len(y_test_original)):
    actual = y_test_original[i]
    predicted = y_pred[i]
    success_percentage = 100 - (abs(actual - predicted) / actual * 100)
    print(f"Ay: {i+1}, Gerçek: {actual:.2f}, Tahmin: {predicted:.2f}, Başarı: %{success_percentage:.2f}")

