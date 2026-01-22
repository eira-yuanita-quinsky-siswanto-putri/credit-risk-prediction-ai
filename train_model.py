import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================================================
# 1. MEMUAT DATASET
# ======================================================
df = pd.read_csv("credit_risk_dataset.csv")

# ======================================================
# 2. PREPROCESSING / ENCODING
# ======================================================

# Encoding kepemilikan rumah
df["person_home_ownership"] = df["person_home_ownership"].map({
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "OTHER": 3
})

# Encoding grade pinjaman
df["loan_grade"] = df["loan_grade"].map({
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
})

# ======================================================
# 3. SELEKSI FITUR & TARGET
# ======================================================
X = df[
    [
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_grade",
        "person_home_ownership"
    ]
]

y = df["loan_status"]

# ======================================================
# 4. SPLIT DATA
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# 5. NORMALISASI DATA
# ======================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================
# 6. TRAIN MODEL (RANDOM FOREST)
# ======================================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ======================================================
# 7. EVALUASI MODEL
# ======================================================
y_pred = model.predict(X_test_scaled)
akurasi = accuracy_score(y_test, y_pred)

print(f"Akurasi Model: {akurasi:.2%}")

# ======================================================
# 8. SIMPAN MODEL & SCALER
# ======================================================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model dan scaler berhasil disimpan.")
