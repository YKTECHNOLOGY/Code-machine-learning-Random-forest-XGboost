from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Path ke file di Google Drive
file_path = "/content/drive/My Drive/laptop_prices.csv"

# Membaca dataset
df = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama
df.head()


# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mengubah data kategorikal menjadi numerik
label_encoders = {}
categorical_columns = ["Brand", "Processor", "Storage", "GPU", "Resolution", "Operating System"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Simpan encoder jika ingin decoding kembali

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=["Price ($)"])  # Semua kolom kecuali harga
y = df["Price ($)"]


# Split data menjadi 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cek ukuran data
print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")


# Inisialisasi model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training model
rf_model.fit(X_train, y_train)

# Prediksi harga di data uji
y_pred = rf_model.predict(X_test)

# Evaluasi Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
