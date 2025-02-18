#!pip install xgboost  # Install XGBoost jika belum ada

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load Dataset
url = "/content/drive/My Drive/laptop_prices.csv"  # Ganti dengan path file Anda
df = pd.read_csv(url)

# Label Encoding untuk data kategorikal
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_columns = ["Brand", "Processor", "Storage", "GPU", "Resolution", "Operating System"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=["Price ($)"])
y = df["Price ($)"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Inisialisasi model XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)

# Training model
xgb_model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred_xgb = xgb_model.predict(X_test)


# Evaluasi model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost - MAE: {mae_xgb}")
print(f"XGBoost - MSE: {mse_xgb}")
print(f"XGBoost - R² Score: {r2_xgb}")