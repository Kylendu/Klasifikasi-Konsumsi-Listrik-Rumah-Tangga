# ==========================================
# FILE: app.py (Versi Final + Input Manual)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="UAS Electric Devices", page_icon="⚡", layout="wide")

# --- 1. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Pastikan nama file sesuai dengan yang ada di folder laptop Anda
        model = joblib.load('rocket_electric_model_final.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# --- 2. HEADER APLIKASI ---
st.title("⚡ Sistem Klasifikasi Konsumsi Listrik (UAS)")
st.markdown("""
Aplikasi ini mendemonstrasikan kemampuan model **ROCKET** dalam mengklasifikasikan 
7 jenis perangkat elektronik berdasarkan pola data 2 menit (96 titik waktu).
""")

# --- 3. SIDEBAR (INPUT) ---
st.sidebar.header("🔧 Panel Pengujian")
input_source = st.sidebar.radio("Sumber Data:", 
                                ["Input Manual (Copy-Paste)", 
                                 "Upload CSV Manual Test", 
                                 "Input Data Dummy"])

class_map = {
    1: 'Kettle', 2: 'Microwave', 3: 'Toaster',
    4: 'Dishwasher', 5: 'Tumble Dryer', 
    6: 'Washing Machine', 7: 'Fridge'
}

# Variabel penampung data
input_series = None
true_label = None

# =========================================================
# LOGIKA 1: INPUT MANUAL (COPY-PASTE) -> FITUR BARU
# =========================================================
if input_source == "Input Manual (Copy-Paste)":
    st.sidebar.info("Paste deretan angka (dipisah koma).")
    st.sidebar.markdown("""
    **Format:**
    - **96 Angka:** Hanya Data (Sistem akan memprediksi tanpa kunci jawaban).
    - **97 Angka:** Data + Label di akhir (Sistem akan mengoreksi prediksi).
    """)
    
    # Text Area untuk Paste
    raw_text = st.sidebar.text_area("Paste Data di Sini:", height=150,
                                    placeholder="-0.574, 1.723, ... (Total 96 angka)")
    
    if raw_text:
        try:
            # Bersihkan spasi/enter dan split berdasarkan koma
            # Menghapus karakter titik di akhir jika user tidak sengaja copy-paste kalimat
            clean_text = raw_text.replace('\n', '').strip().rstrip('.')
            str_values = clean_text.split(',')
            
            # Konversi ke float
            float_values = [float(x) for x in str_values if x.strip() != '']
            
            # Cek Panjang Data
            if len(float_values) == 96:
                input_series = np.array(float_values)
                # UBAHAN DI SINI: Mengubah Warning menjadi Success
                st.sidebar.success("✅ Data Valid: 96 titik terdeteksi.")
                st.sidebar.info("ℹ️ Mode: Prediksi Blind (Tanpa Kunci Jawaban)")
                
            elif len(float_values) == 97:
                input_series = np.array(float_values[:96]) # Ambil 96 awal
                true_label = int(float_values[96])         # Ambil angka terakhir sbg Label
                st.sidebar.success(f"✅ Data Valid: 96 titik + Label Kelas {true_label}")
            else:
                st.sidebar.error(f"⚠️ Error: Jumlah data harus 96 atau 97. Terdeteksi: {len(float_values)}")
                
        except ValueError:
            st.sidebar.error("⚠️ Error: Pastikan semua input adalah angka dan dipisahkan koma.")

# =========================================================
# LOGIKA 2: UPLOAD CSV MANUAL
# =========================================================
elif input_source == "Upload CSV Manual Test":
    st.sidebar.info("Gunakan file 'data_test_manual_5490.csv'.")
    uploaded_file = st.sidebar.file_uploader("Upload File CSV", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            row_idx = st.sidebar.number_input("Pilih Index Baris", 
                                            min_value=0, max_value=len(df)-1, value=0)
            
            row_data = df.iloc[row_idx].values
            
            if len(row_data) == 97:
                input_series = row_data[:96]
                true_label = int(row_data[96])
                st.sidebar.success(f"Baris {row_idx} Terpilih (Kelas {true_label})")
            elif len(row_data) == 96:
                input_series = row_data
            else:
                st.error(f"Format data salah. Kolom: {len(row_data)}")
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# LOGIKA 3: DATA DUMMY
# =========================================================
elif input_source == "Input Data Dummy":
    alat = st.sidebar.selectbox("Pilih Simulasi Alat:", ["Kettle (Spike)", "Washing Machine (Cycle)"])
    if st.sidebar.button("Generate"):
        dummy = np.zeros(96)
        if "Kettle" in alat:
            dummy[80:85] = 8.0 
            true_label = 1
        else:
            dummy[10:90] = np.random.normal(0, 0.5, 80)
            dummy[30:50] += 2
            dummy[70:85] += 6
            true_label = 6
        input_series = dummy

# =========================================================
# TAMPILAN UTAMA (VISUALISASI & PREDIKSI)
# =========================================================
col1, col2 = st.columns([2, 1])

if input_series is not None:
    # A. Visualisasi Grafik
    with col1:
        st.subheader("Visualisasi Sinyal Listrik")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(input_series, color='#007acc', linewidth=2)
        ax.set_title("Pola Konsumsi Daya (96 Time-Steps)")
        ax.set_ylabel("Power (Normalized)")
        ax.set_xlabel("Waktu")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # B. Hasil Prediksi
    with col2:
        st.subheader("Hasil Diagnosa AI")
        
        if st.button("Jalankan Analisis ROCKET", use_container_width=True):
            if model:
                with st.spinner("Sedang mengekstraksi 10.000 fitur..."):
                    try:
                        # Format input untuk sktime
                        X_input = pd.DataFrame({'dim_0': [pd.Series(input_series)]})
                        
                        # Prediksi
                        pred_class = model.predict(X_input)[0]
                        pred_name = class_map.get(pred_class, "Unknown")
                        
                        # Tampilkan Hasil
                        if true_label is not None:
                            if pred_class == true_label:
                                st.success("✅ PREDIKSI BENAR")
                            else:
                                st.error("❌ PREDIKSI SALAH")
                        elif true_label is None:
                            st.info("ℹ️ Hasil prediksi (Data tanpa label)")

                        st.metric("Prediksi Model", f"{pred_name}")
                        st.metric("ID Kelas", f"{pred_class}")
                        
                        if true_label is not None:
                            st.info(f"Label Asli (Kunci): {class_map.get(true_label)}")
                            
                    except Exception as e:
                        st.error(f"Error Prediksi: {e}")
                        st.warning("Cek versi library NumPy/Scikit-learn Anda.")
            else:
                st.error("Model belum dimuat.")
else:
    st.info("👈 Silakan pilih metode input di Sidebar.")