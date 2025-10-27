import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="AI Vision", page_icon="🕊️", layout="wide")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
    body {
        background-color: #f9f0f6;
    }
    .stApp {
        background-color: #f9f0f6;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4 {
        color: #4b4453;
        text-align: center;
    }

    .center {
        text-align: center;
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }

    .pink-btn {
        background-color: #e91e63;
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }

    .pink-btn:hover {
        background-color: #d81b60;
        transform: scale(1.05);
        color: white;
    }

    .choice-box {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 40px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        color: #555;
        font-size: 0.9em;
    }

</style>
""", unsafe_allow_html=True)

# ==========================
# Navigasi Halaman
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# ==========================
# Halaman 1: Beranda
# ==========================
if st.session_state.page == "home":
    st.image("https://cdn.pixabay.com/photo/2016/11/18/12/03/flamingo-1835643_1280.jpg", use_container_width=False)
    st.markdown("<h2>Selamat Datang di</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#e91e63;'>AI Vision</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center'>Platform AI canggih untuk mendeteksi objek dan mengklasifikasi jenis burung dalam gambar dengan teknologi deep learning terdepan.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'><h4>🔍 Deteksi Objek</h4><p>Identifikasi dan lokalisasi objek burung dalam gambar dengan presisi tinggi.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><h4>🪶 Klasifikasi Gambar</h4><p>Klasifikasikan jenis burung berdasarkan spesies dan karakteristiknya.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><h4>🎯 Akurasi Tinggi</h4><p>Hasil analisis dengan tingkat akurasi dan confidence score tinggi.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='center'><br><button class='pink-btn' onclick='window.location.reload()'>Mulai Sekarang</button></div>", unsafe_allow_html=True)
    if st.button("🚀 Mulai Sekarang"):
        go_to("pilih")

# ==========================
# Halaman 2: Pilih Jenis Analisis
# ==========================
elif st.session_state.page == "pilih":
    st.markdown("<h2>Pilih Jenis Analisis</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#e91e63;'>AI Vision</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center'>Pilih mode analisis yang ingin Anda gunakan untuk menganalisis gambar burung.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧭 Deteksi Objek", use_container_width=True):
            go_to("deteksi")
        st.markdown("<p class='center'>Temukan dan lokalisasi objek dalam gambar dengan bounding box.</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🪶 Klasifikasi Gambar", use_container_width=True):
            go_to("klasifikasi")
        st.markdown("<p class='center'>Identifikasi jenis dan spesies burung dengan detail lengkap.</p>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali", use_container_width=False):
        go_to("home")

# ==========================
# Halaman 3: Upload & Analisis
# ==========================
elif st.session_state.page in ["deteksi", "klasifikasi"]:
    mode = st.session_state.page
    st.markdown(f"<h2>Selamat Datang, Pengguna!</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='center'>Upload gambar burung untuk memulai analisis AI ({'Deteksi Objek' if mode == 'deteksi' else 'Klasifikasi Gambar'})</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
    with col2:
        st.info("Upload gambar dan klik 'Analisis AI' untuk melihat hasil")

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
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                st.success(f"✅ Hasil Prediksi: Kelas {class_index}")
                st.write("Confidence:", float(np.max(prediction)))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali"):
        go_to("pilih")

# ==========================
# Footer
# ==========================
st.markdown("<div class='footer'>© 2025 AI Vision | Dibuat oleh Khaira Putri Syalaisa</div>", unsafe_allow_html=True)
