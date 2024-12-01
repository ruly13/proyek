import streamlit as st
import pickle
import numpy as np
import sklearn

# Memuat model prediksi
with open('anemia_pred.pkl', 'rb') as f:
    model = pickle.load(f)

# Memuat scaler untuk normalisasi data
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fungsi untuk melakukan prediksi
def predict(jenis_kelamin, fitur1, fitur2, fitur3, fitur4):
    jenis_kelamin_encoded = np.array([0, 0])
    if jenis_kelamin == 'Laki-Laki':
        jenis_kelamin_encoded[1] = 1
    else:
        jenis_kelamin_encoded[0] = 1
    jenis_kelamin_encoded = jenis_kelamin_encoded.reshape(1, -1)
    
    fitur = np.array([[fitur1, fitur2, fitur3, fitur4]]).astype(float)
    fitur_ternormalisasi = scaler.transform(fitur)
    
    fitur_final = np.hstack((jenis_kelamin_encoded, fitur_ternormalisasi))
    
    prediksi = model.predict(fitur_final)
    return prediksi[0]

# Simbol jenis kelamin
simbol_laki = "\u2642"
simbol_perempuan = "\u2640"

# Judul aplikasi
st.title("🌟 Prediksi Anemia dengan Klasifikasi 🌟")

st.markdown("""
<style>
.stRadio > div {flex-direction:row;}
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    padding: 10px 24px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
div.stButton > button:first-child:hover {
    background-color: #45a049;
}
.stSlider {width: 100%;}

.result-box {
    padding: 10px;
    border-radius: 5px;
    color: #fff;
    font-weight: bold;
    text-align: center;
}
.result-yes {
    background-color: #c9463c; 
}
.result-no {
    background-color: #1a691e; 
}
</style>
""", unsafe_allow_html=True)

st.markdown("### Silakan masukkan data berikut:")

# Input data
col1, col2 = st.columns(2)

with col1:
    jenis_kelamin = st.radio("Jenis Kelamin", [f"Laki-Laki {simbol_laki}", f"Perempuan {simbol_perempuan}"])

col1, col2 = st.columns(2)

with col1:
    fitur1 = st.slider(":red[Piksel Merah]", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Masukkan intensitas piksel merah untuk menganalisis kadar hemoglobin.")
    fitur2 = st.slider(":green[Piksel Hijau]", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Masukkan nilai piksel hijau untuk mendeteksi fitur tertentu dalam sampel darah.")

with col2:
    fitur3 = st.slider(":blue[Piksel Biru]", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Masukkan data piksel biru untuk meningkatkan akurasi analisis fitur.")
    fitur4 = st.slider("Hb", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Masukkan kadar hemoglobin untuk prediksi risiko anemia.")

# Tombol prediksi
if st.button("Prediksi"):
    hasil = predict(jenis_kelamin.split()[0], fitur1, fitur2, fitur3, fitur4)
    if hasil == 1:
        st.markdown('<div class="result-box result-yes">Hasil prediksi: Ya. Ada risiko tinggi anemia.</div>', unsafe_allow_html=True)
    else:
         st.markdown('<div class="result-box result-no">Hasil prediksi: Tidak. Risiko anemia rendah.</div>', unsafe_allow_html=True)

# Referensi
st.markdown("""
<div class="references">
    <h3>Referensi</h3>
    <ul>
        <li><a href="https://www.sciencedirect.com/science/article/pii/S2590093523000322#fig4" target="_blank">Referensi 1</a></li>
        <li><a href="https://iopscience.iop.org/article/10.1088/1757-899X/420/1/012101/pdf" target="_blank">Referensi 2</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)
