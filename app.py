# ==========================================
# FILE: app.py (FINAL - CLEAN VERSION)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# SESSION STATE INIT
# ==========================================
if "input_series" not in st.session_state:
    st.session_state.input_series = None

if "true_label" not in st.session_state:
    st.session_state.true_label = None

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="UAS Electric Devices",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("rocket_electric_model_final.pkl")

model = load_model()

# ==========================================
# HEADER
# ==========================================
st.title("⚡ Sistem Klasifikasi Konsumsi Listrik (UAS)")
st.markdown("""
Aplikasi ini mendemonstrasikan kemampuan model **ROCKET**
dalam mengklasifikasikan **7 jenis perangkat listrik**
berdasarkan pola deret waktu **96 titik**.
""")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Panel Pengujian")

input_source = st.sidebar.radio(
    "Sumber Data:",
    ["Input Manual (Copy-Paste)", "Input Data Dummy"]
)

class_map = {
    1: "Kettle (Ketel)",
    2: "Immersion Heater",
    3: "Washing Machine",
    4: "Dishwasher",
    5: "Oven / Cooker",
    6: "Fridge / Freezer",
    7: "Computer"
}

# ==========================================
# MODE 1: INPUT MANUAL
# ==========================================
if input_source == "Input Manual (Copy-Paste)":
    st.sidebar.info("Paste 96 angka atau 96 + 1 label di akhir.")
    raw_text = st.sidebar.text_area("Paste Data:", height=150)

    if raw_text:
        try:
            values = [
                float(x) for x in raw_text.replace("\n", "")
                .strip().rstrip(".").split(",") if x.strip() != ""
            ]

            if len(values) == 96:
                st.session_state.input_series = np.array(values)
                st.session_state.true_label = None
                st.sidebar.success("96 titik terdeteksi (Blind Prediction)")

            elif len(values) == 97:
                st.session_state.input_series = np.array(values[:96])
                st.session_state.true_label = int(values[96])
                st.sidebar.success(
                    f"96 titik + Label {st.session_state.true_label}"
                )

            else:
                st.sidebar.error(f"Jumlah data salah: {len(values)}")

        except ValueError:
            st.sidebar.error("Format harus berupa angka dipisah koma.")

# ==========================================
# MODE 2: DATA DUMMY
# ==========================================
elif input_source == "Input Data Dummy":
    st.sidebar.info("Data berhasil di Generate")

    if st.sidebar.button("Generate Data"):
        try:
            df = pd.read_csv("data_test_manual_5490.csv", header=None)
            row = df.sample(1).iloc[0].values

            st.session_state.input_series = row[:96]
            st.session_state.true_label = int(row[96])

            st.sidebar.success(
                f"Data Dummy Aktif (Label: {st.session_state.true_label})"
            )

        except Exception as e:
            st.sidebar.error(f"Gagal memuat CSV: {e}")

# ==========================================
# READ FROM SESSION STATE
# ==========================================
input_series = st.session_state.input_series
true_label = st.session_state.true_label

# ==========================================
# VISUALISASI & PREDIKSI
# ==========================================
col1, col2 = st.columns([2, 1])

if input_series is not None:
    with col1:
        st.subheader("Visualisasi Sinyal Listrik")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(input_series, linewidth=2)
        ax.set_title("Pola Konsumsi Daya (96 Time-Steps)")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Power (Normalized)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Hasil Diagnosa AI")

        if st.button("Jalankan Analisis ROCKET", use_container_width=True):
            with st.spinner("Ekstraksi fitur & klasifikasi..."):
                try:
                    X_input = pd.DataFrame({
                        "dim_0": [pd.Series(input_series)]
                    })

                    pred_class = model.predict(X_input)[0]
                    pred_name = class_map.get(pred_class, "Unknown")

                    if true_label is not None:
                        if pred_class == true_label:
                            st.success("PREDIKSI BENAR")
                        else:
                            st.error("PREDIKSI SALAH")

                    st.metric("Prediksi Model", pred_name)
                    st.metric("ID Kelas", pred_class)

                    if true_label is not None:
                        st.info(
                            f"Label Asli: {class_map.get(true_label)}"
                        )

                except Exception as e:
                    st.error(f"Error Prediksi: {e}")

else:
    st.info("Pilih metode input di Sidebar untuk memulai.")

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    "<hr><center><small>UAS Time Series Classification · ROCKET · Streamlit</small></center>",
    unsafe_allow_html=True
)