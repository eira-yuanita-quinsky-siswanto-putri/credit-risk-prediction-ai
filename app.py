import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ======================================================
# MEMUAT MODEL & SCALER
# ======================================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ======================================================
# MEMUAT DATASET
# ======================================================
DATA_PATH = "credit_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Prediksi Risiko Kredit",
    layout="wide"
)

st.title("📊 Sistem Prediksi Risiko Kredit")
st.caption(
    "Sistem Pendukung Keputusan Berbasis Artificial Intelligence (Random Forest) "
    "untuk memprediksi risiko gagal bayar calon peminjam kredit"
)

st.divider()

# ======================================================
# 📝 INPUT DATA CALON PEMINJAM
# ======================================================
st.subheader("📝 Input Data Calon Peminjam")

col1, col2, col3 = st.columns(3)

with col1:
    usia = st.number_input(
        "Usia (Tahun)",
        min_value=18,
        max_value=70,
        help="Usia calon peminjam dalam tahun"
    )

    pendapatan = st.number_input(
        "Pendapatan Tahunan (USD)",
        min_value=0,
        help="Total pendapatan calon peminjam dalam satu tahun (Dollar Amerika)"
    )

with col2:
    jumlah_pinjaman = st.number_input(
        "Jumlah Pinjaman (USD)",
        min_value=0,
        help="Total nilai pinjaman yang diajukan (Dollar Amerika)"
    )

    suku_bunga = st.number_input(
        "Suku Bunga (%)",
        min_value=0.0,
        help="Suku bunga pinjaman dalam persentase per tahun"
    )

with col3:
    grade_pinjaman = st.selectbox(
        "Grade Pinjaman",
        ["A", "B", "C", "D", "E", "F", "G"],
        help="Penilaian kualitas pinjaman (A = sangat baik, G = sangat berisiko)"
    )

    kepemilikan_rumah_label = st.selectbox(
        "Status Kepemilikan Rumah",
        [
            "Sewa (Belum memiliki rumah)",
            "KPR / Kredit Pemilikan Rumah",
            "Milik Sendiri",
            "Lainnya (Menumpang / Rumah Dinas)"
        ],
        help="Status tempat tinggal calon peminjam"
    )

# ======================================================
# KETERANGAN UNTUK USER
# ======================================================
st.caption(
    "📌 **Keterangan Satuan & Istilah:**\n"
    "- Pendapatan Tahunan dan Jumlah Pinjaman menggunakan **USD (Dollar Amerika)** sesuai dataset\n"
    "- Suku bunga dinyatakan dalam **persentase (%) per tahun**\n"
    "- Grade A menunjukkan risiko rendah, Grade G menunjukkan risiko tinggi"
)

st.divider()

# ======================================================
# MAPPING & ENCODING
# ======================================================
peta_grade = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

peta_rumah_label = {
    "Sewa (Belum memiliki rumah)": "RENT",
    "KPR / Kredit Pemilikan Rumah": "MORTGAGE",
    "Milik Sendiri": "OWN",
    "Lainnya (Menumpang / Rumah Dinas)": "OTHER"
}

peta_rumah = {
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "OTHER": 3
}

grade_encoded = peta_grade[grade_pinjaman]
kepemilikan_rumah = peta_rumah_label[kepemilikan_rumah_label]
rumah_encoded = peta_rumah[kepemilikan_rumah]

# ======================================================
# 🔍 PREDIKSI & SIMPAN DATA
# ======================================================
if st.button("🔍 Prediksi Risiko & Simpan Data"):

    data_model = np.array([[
        usia,
        pendapatan,
        jumlah_pinjaman,
        suku_bunga,
        grade_encoded,
        rumah_encoded
    ]])

    data_scaled = scaler.transform(data_model)
    hasil_prediksi = model.predict(data_scaled)[0]
    probabilitas = model.predict_proba(data_scaled)[0][1]

    label_hasil = "Risiko Tinggi" if hasil_prediksi == 1 else "Risiko Rendah"

    if hasil_prediksi == 1:
        st.error(f"⚠️ **{label_hasil}** (Probabilitas: {probabilitas:.2%})")
    else:
        st.success(f"✅ **{label_hasil}** (Probabilitas: {probabilitas:.2%})")

    # ==================================================
    # SIMPAN DATA KE DATASET
    # ==================================================
    data_baru = {
        "person_age": usia,
        "person_income": pendapatan,
        "loan_amnt": jumlah_pinjaman,
        "loan_int_rate": suku_bunga,
        "loan_grade": grade_pinjaman,
        "person_home_ownership": kepemilikan_rumah,
        "loan_status": hasil_prediksi,
        "hasil_prediksi": label_hasil,
        "probabilitas_risiko": round(probabilitas, 4)
    }

    df = pd.concat([df, pd.DataFrame([data_baru])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    st.success("📌 Data calon peminjam berhasil disimpan ke dataset")

st.divider()

# ======================================================
# 📁 TABEL DATASET
# ======================================================
st.subheader("📁 Data Kredit Nasabah")
st.dataframe(df, use_container_width=True)

st.divider()

# ======================================================
# 📈 DISTRIBUSI RISIKO
# ======================================================
st.subheader("📈 Distribusi Risiko Kredit")

jumlah_risiko = df["loan_status"].value_counts()

fig, ax = plt.subplots()
ax.bar(jumlah_risiko.index.astype(str), jumlah_risiko.values)
ax.set_xlabel("Status Kredit (0 = Lancar, 1 = Gagal Bayar)")
ax.set_ylabel("Jumlah Data")
st.pyplot(fig)

st.divider()

# ======================================================
# 📊 DISTRIBUSI KATEGORI RISIKO KREDIT
# ======================================================
st.subheader("📊 Distribusi Kategori Risiko Kredit")

kategori_risiko = df["hasil_prediksi"].value_counts()

fig_kat, ax_kat = plt.subplots()
ax_kat.pie(
    kategori_risiko.values,
    labels=kategori_risiko.index,
    autopct="%1.1f%%",
    startangle=90
)
ax_kat.axis("equal")
st.pyplot(fig_kat)

st.divider()

# ======================================================
# 🧠 FEATURE IMPORTANCE
# ======================================================
st.subheader("🧠 Faktor yang Mempengaruhi Risiko Kredit")

nama_fitur = [
    "Usia",
    "Pendapatan Tahunan (USD)",
    "Jumlah Pinjaman (USD)",
    "Suku Bunga",
    "Grade Pinjaman",
    "Kepemilikan Rumah"
]

tingkat_pengaruh = model.feature_importances_

fig2, ax2 = plt.subplots()
ax2.barh(nama_fitur, tingkat_pengaruh)
ax2.set_xlabel("Tingkat Pengaruh")
st.pyplot(fig2)

st.info("Semakin besar nilai, semakin besar pengaruh fitur terhadap risiko gagal bayar.")

# ======================================================
# 🧠 INTERPRETASI TINGKAT KEPENTINGAN FITUR
# ======================================================
st.subheader("🧠 Tingkat Kepentingan Fitur (Urutan Pengaruh)")

df_importance = pd.DataFrame({
    "Fitur": nama_fitur,
    "Nilai Kepentingan": tingkat_pengaruh
}).sort_values(by="Nilai Kepentingan", ascending=True)

fig_imp, ax_imp = plt.subplots()
ax_imp.barh(
    df_importance["Fitur"],
    df_importance["Nilai Kepentingan"]
)
ax_imp.set_xlabel("Nilai Kepentingan")
ax_imp.set_ylabel("Fitur")
st.pyplot(fig_imp)

st.caption(
    "Grafik ini menampilkan urutan fitur dari yang paling rendah hingga paling tinggi "
    "dalam mempengaruhi keputusan risiko kredit pada model Random Forest."
)
