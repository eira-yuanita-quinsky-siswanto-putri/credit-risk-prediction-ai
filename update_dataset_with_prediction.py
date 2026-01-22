import pandas as pd
import pickle

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ======================================================
# LOAD DATASET
# ======================================================
df = pd.read_csv("credit_risk_dataset.csv")

# ======================================================
# COPY DATA UNTUK PREDIKSI (AMAN)
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
].copy()

# ======================================================
# ENCODING (SAMA PERSIS SAAT TRAINING)
# ======================================================
X["loan_grade"] = X["loan_grade"].map({
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7
})

X["person_home_ownership"] = X["person_home_ownership"].map({
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "OTHER": 3
})

# ======================================================
# SCALING
# ======================================================
X_scaled = scaler.transform(X)

# ======================================================
# PREDIKSI
# ======================================================
df["loan_status"] = model.predict(X_scaled)
df["probabilitas_risiko"] = model.predict_proba(X_scaled)[:, 1]

df["hasil_prediksi"] = df["loan_status"].apply(
    lambda x: "Risiko Tinggi" if x == 1 else "Risiko Rendah"
)

# ======================================================
# SIMPAN KEMBALI
# ======================================================
df.to_csv("credit_risk_dataset.csv", index=False)

print("✅ Dataset berhasil diperbarui dengan hasil prediksi & probabilitas")
