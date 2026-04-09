import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os

# CSV dosyasını oku
data_path = "data/"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

if not csv_files:
    print("data/ klasöründe CSV dosyası bulunamadı!")
    exit()

# İlk CSV dosyasını oku
df = pd.read_csv(os.path.join(data_path, csv_files[0]))
print(f"Veri seti yüklendi: {csv_files[0]}")
print(f"Veri seti boyutu: {df.shape}")
print(f"Kolonlar: {list(df.columns)}")

# 'Machine failure' kolonunu kontrol et
if 'Machine failure' not in df.columns:
    print("'Machine failure' kolonu bulunamadı! Mevcut kolonlar:", list(df.columns))
    exit()

# Hedef değişken ve özellikleri ayır
target = 'Machine failure'
# Sayısal kolonları ve sensör verilerini feature olarak seç
sensor_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != target]
features = sensor_columns

print(f"\nÖzellikler: {features}")
print(f"Hedef: {target}")

# Eksik verileri kontrol et ve doldur
print(f"\nEksik veri sayısı:\n{df[features + [target]].isnull().sum()}")

# Eksik verileri medyan ile doldur
for col in features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Hedef değişkendeki eksik verileri kaldır
df = df.dropna(subset=[target])

print(f"\nTemizlenmiş veri seti boyutu: {df.shape}")
print(f"Sınıf dağılımı:\n{df[target].value_counts()}")

# Özellikleri ve hedefi ayır
X = df[features]
y = df[target]

# Veri setini train-test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Sınıf ağırlıklarını hesapla
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"\nSınıf ağırlıkları: {class_weight_dict}")

# CatBoost modelini oluştur ve eğit
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='F1',
    class_weights=[class_weight_dict[0], class_weight_dict[1]],
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

print("\nModel eğitimi başlıyor...")
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=100
)

# Modeli kaydet
model.save_model("models/catboost_model.cbm")
print("\nModel kaydedildi: models/catboost_model.cbm")

# Tahminler yap
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Performans metrikleri
print("\n=== MODEL PERFORMANSI ===")
print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Skoru: {f1_score(y_test, y_pred):.4f}")

print("\n=== SINIFLANDIRMA RAPORU ===")
print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))

# Özellik önemliliği
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== ÖZELLİK ÖNEMLİLİĞİ ===")
print(feature_importance.head(10))

print(f"\nModel başarıyla eğitildi ve kaydedildi!")
print(f"Model dosyası: models/catboost_model.cbm")
