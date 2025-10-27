import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# =============== CONFIGURASI DASAR ===============
st.set_page_config(page_title="AI Vision", layout="wide")

@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# =============== CSS UNTUK DESAIN ===============
st.markdown("""
<style>
body {background-color:#fdf4f8;}
.stApp {background-color:#fdf4f8; font-family:'Poppins',sans-serif;}

h1,h2,h3 {text-align:center; color:#2d2b30;}
p {text-align:center; color:#444;}

.btn-main {
    background-color:#e91e63; color:white; border:none;
    padding:0.8rem 2rem; border-radius:12px; font-weight:600;
    transition:0.3s; display:inline-block;
}
.btn-main:hover {background-color:#d81b60; transform:scale(1.05);}

.card {
    background:white; border-radius:15px; padding:1rem;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
    text-align:center; transition:0.3s;
}
.card:hover {transform:scale(1.03);}

.choice {
    width:100%; background:white; border-radius:15px;
    padding:2rem; text-align:center; cursor:pointer;
    box-shadow:0 4px 8px rgba(0,0,0,0.05);
    transition:0.3s; border:2px solid transparent;
}
.choice:hover {transform:scale(1.02);}
.choice-deteksi {border-color:#f48fb1;}
.choice-klasifikasi {border-color:#b39ddb;}
.footer {text-align:center; margin-top:40px; color:#555;}
</style>
""", unsafe_allow_html=True)

# =============== NAVIGASI ANTAR HALAMAN ===============
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(page):
    st.session_state.page = page

# =====================================================
# =============== HALAMAN 1: BERANDA ==================
# =====================================================
if st.session_state.page == "home":
    st.image("https://cdn.pixabay.com/photo/2016/11/18/12/03/flamingo-1835643_1280.jpg", use_container_width=False)
    st.markdown("<h2>Selamat Datang di</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#e91e63;'>AI Vision</h1>", unsafe_allow_html=True)
    st.markdown("<p>Platform AI canggih untuk mendeteksi objek dan mengklasifikasi jenis burung dalam gambar dengan teknologi deep learning terdepan.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'><h4>🔍 Deteksi Objek</h4><p>Identifikasi dan lokalisasi objek burung dalam gambar dengan presisi tinggi.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><h4>🪶 Klasifikasi Gambar</h4><p>Mengklasifikasi jenis burung dengan detail spesies dan karakteristik.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><h4>🎯 Akurasi Tinggi</h4><p>Hasil analisis dengan tingkat akurasi tinggi dan confidence score.</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
    if st.button("Mulai Sekarang 🚀", use_container_width=False):
        goto("pilih")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:30px;'>
        <b>95%</b> Akurasi &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>200+</b> Spesies Burung &nbsp;&nbsp; | &nbsp;&nbsp;
        <b><2s</b> Waktu Proses
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# =============== HALAMAN 2: PILIH ANALISIS ===========
# =====================================================
elif st.session_state.page == "pilih":
    st.markdown("<h2>Pilih Jenis Analisis</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#e91e63;'>AI Vision</h1>", unsafe_allow_html=True)
    st.markdown("<p>Pilih mode analisis yang ingin Anda gunakan untuk menganalisis gambar burung.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧭 Deteksi Objek", use_container_width=True):
            goto("deteksi")
        st.markdown("<p>Temukan dan lokalisasi objek dalam gambar dengan bounding box.</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🪶 Klasifikasi Gambar", use_container_width=True):
            goto("klasifikasi")
        st.markdown("<p>Identifikasi jenis dan spesies burung dengan detail lengkap.</p>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align:center; margin-top:40px;'>Teknologi AI yang Digunakan</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; justify-content:center; gap:60px;'>
        <div><b>Deep Learning</b><br>Menggunakan CNN untuk analisis gambar cepat dan akurat.</div>
        <div><b>Computer Vision</b><br>Teknologi visual untuk memahami konten gambar.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Kembali", use_container_width=False):
        goto("home")

# =====================================================
# =============== HALAMAN 3: ANALISIS ================
# =====================================================
elif st.session_state.page in ["deteksi", "klasifikasi"]:
    mode = st.session_state.page
    st.markdown(f"<h2>Selamat Datang, Pengguna!</h2>", unsafe_allow_html=True)
    st.markdown(f"<p>Upload gambar burung untuk memulai analisis AI</p>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
        st.caption("Format: JPG, PNG, JPEG (Max 10MB)")
    with col_b:
        st.info("Upload gambar dan klik 'Analisis AI' untuk melihat hasil di sini")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        if st.button("🔎 Analisis AI"):
            if mode == "deteksi":
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
            else:
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                pred = classifier.predict(img_array)
                st.success(f"Hasil Prediksi: {np.argmax(pred)}")
                st.write("Confidence:", float(np.max(pred)))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali"):
        goto("pilih")

try:
    yolo_model = YOLO("model/best.pt")
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {e}")

# =====================================================
# =============== FOOTER ==============================
# =====================================================
st.markdown("<div class='footer'>© 2025 AI Vision | Dibuat oleh Khaira Putri Syalaisa</div>", unsafe_allow_html=True)
