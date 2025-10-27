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
st.set_page_config(page_title="Deteksi & Klasifikasi Gambar", layout="wide")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
    body {
        background-color: #f5e6f1;
        color: #4B4453;
        font-family: 'Poppins', sans-serif;
    }

    .main-title {
        text-align: center;
        color: #4B4453;
        font-size: 48px;
        margin-top: 100px;
        font-weight: 600;
    }

    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #6a5d73;
        margin-bottom: 40px;
    }

    .button-container {
        text-align: center;
    }

    .pink-button {
        background-color: #ec407a;
        color: white;
        padding: 15px 40px;
        border: none;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 500;
        margin: 15px;
        cursor: pointer;
        transition: 0.3s;
    }

    .pink-button:hover {
        background-color: #d81b60;
        transform: scale(1.05);
    }

    .back-button {
        background-color: transparent;
        color: #ec407a;
        border: 2px solid #ec407a;
        border-radius: 25px;
        padding: 8px 20px;
        cursor: pointer;
        font-size: 16px;
        margin-top: 20px;
    }

    .card {
        background-color: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Navigasi Antar Halaman
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(page):
    st.session_state.page = page

# ==========================
# 1️⃣ HALAMAN UTAMA
# ==========================
if st.session_state.page == "home":
    st.markdown("<h1 class='main-title'>🧠 Image Detection & Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-text'>Pilih salah satu mode di bawah untuk memulai</p>", unsafe_allow_html=True)

    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    st.button("Klasifikasi Gambar", key="to_class", on_click=lambda: goto("classify"))
    st.button("Deteksi Objek (YOLO)", key="to_detect", on_click=lambda: goto("detect"))
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# 2️⃣ HALAMAN KLASIFIKASI
# ==========================
elif st.session_state.page == "classify":
    st.markdown("<h2 style='text-align:center;'>🧩 Klasifikasi Gambar</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("Sedang melakukan klasifikasi..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            prob = np.max(prediction)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Hasil Prediksi:", class_index)
        st.metric("Probabilitas", f"{prob:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.button("⬅️ Kembali", on_click=lambda: goto("home"), key="back_class")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# 3️⃣ HALAMAN DETEKSI
# ==========================
elif st.session_state.page == "detect":
    st.markdown("<h2 style='text-align:center;'>🎯 Deteksi Objek (YOLO)</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.button("⬅️ Kembali", on_click=lambda: goto("home"), key="back_detect")
    st.markdown("</div>", unsafe_allow_html=True)
