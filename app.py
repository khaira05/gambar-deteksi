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
import os
import io

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="AI Vision", layout="wide", initial_sidebar_state="collapsed")

# ==========================
# CUSTOM CSS (mimic screenshot)
# ==========================
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(180deg, #fff7fb 0%, #f9eef7 40%, #f3e6f3 100%);
        font-family: 'Inter', 'Poppins', sans-serif;
        color: #3b2f3f;
    }

    /* center container override */
    .main-container {
        padding-top: 40px;
        padding-bottom: 60px;
    }

    /* hero text */
    .hero-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin: 0;
        color: #3b2f3f;
    }
    .hero-sub {
        font-size: 16px;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 28px;
        color: #6f5f6f;
    }

    /* big pink buttons in center */
    .btn-row {
        display:flex;
        justify-content:center;
        gap:30px;
        margin-top: 22px;
        margin-bottom: 18px;
    }
    .big-pink {
        background: linear-gradient(90deg, #ff7aa8 0%, #e91e63 100%);
        color: white !important;
        border: none;
        padding: 18px 44px;
        font-size: 18px;
        font-weight: 700;
        border-radius: 28px;
        box-shadow: 0 10px 24px rgba(233,30,99,0.16);
        cursor: pointer;
        transition: transform .14s ease, box-shadow .14s ease;
    }
    .big-pink:hover { transform: translateY(-3px); box-shadow: 0 18px 30px rgba(233,30,99,0.18); }

    /* subtle cards for features */
    .features {
        display:flex;
        justify-content:center;
        gap:20px;
        margin-top: 28px;
    }
    .feature-card {
        background: #ffffffcc;
        border-radius: 14px;
        padding: 18px;
        width: 260px;
        box-shadow: 0 6px 18px rgba(60,40,70,0.06);
        text-align:center;
    }
    .feature-card h4 { margin:6px 0 4px 0; color:#3b2f3f; }
    .feature-card p { margin:0; color:#6f5f6f; font-size:14px; }

    /* upload area */
    .upload-box {
        background: #fff;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 26px rgba(60,40,70,0.06);
    }

    /* result card */
    .result-box {
        background: #fff;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 26px rgba(60,40,70,0.06);
    }

    /* small footer */
    .footer {
        text-align:center;
        color:#7a6a7a;
        margin-top:26px;
        font-size:13px;
    }

    /* hide Streamlit default footer and header look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# LOAD MODELS (CACHED)
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# NAVIGATION STATE
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(page):
    st.session_state.page = page

# ==========================
# LANDING PAGE (identical layout: image + two big pink buttons)
# ==========================
if st.session_state.page == "home":
    # main container
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # hero
    st.markdown("<h1 class='hero-title'>AI Vision</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Aplikasi deteksi objek & klasifikasi gambar — tekan tombol untuk memilih mode</div>", unsafe_allow_html=True)

    # central image (use the sample you had or local asset)
    # if you want exact image from your screenshots, replace the URL with your file path or hosted URL
    st.image("https://cdn.pixabay.com/photo/2016/11/18/12/03/flamingo-1835643_1280.jpg", width=560)

    # two big pink buttons centered
    st.markdown("<div class='btn-row'>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Klasifikasi Gambar", key="btn_class"):
            goto("classify")
    with col_b:
        if st.button("Deteksi Objek", key="btn_detect"):
            goto("detect")
    st.markdown("</div>", unsafe_allow_html=True)

    # small feature cards below (mimic screenshot)
    st.markdown(
        """
        <div class='features'>
          <div class='feature-card'><h4>🔍 Deteksi Cepat</h4><p>Bounding box & confidence</p></div>
          <div class='feature-card'><h4>🪶 Klasifikasi Akurat</h4><p>Model terlatih untuk banyak kelas</p></div>
          <div class='feature-card'><h4>⚡ Respon Kilat</h4><p>Proses cepat & ringan</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close main-container
    st.markdown("<div class='footer'>© 2025 AI Vision — Khaira Putri Syalaisa</div>", unsafe_allow_html=True)

else:
    # ===============================
    # Header Dinamis (Judul Halaman)
    # ===============================
    if st.session_state.page == "detect":
        mode = "Deteksi Objek (YOLO)"
        icon = "🧠"
        desc = "Unggah gambar untuk mendeteksi objek menggunakan model YOLO."
    else:
        mode = "Klasifikasi Gambar"
        icon = "📊"
        desc = "Unggah gambar untuk memprediksi kelas menggunakan model klasifikasi."

    st.markdown(f"""
        <div style="
            text-align:center;
            padding: 30px 0 10px 0;
            background: linear-gradient(135deg, #1E293B, #334155);
            color: white;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <h1 style="font-size:36px; margin-bottom:8px;">{icon} {mode}</h1>
            <p style="font-size:16px; color:#CBD5E1;">{desc}</p>
        </div>
    """, unsafe_allow_html=True)

    # ===============================
    # Upload Section
    # ===============================
    st.markdown("""
        <div style="
            background-color:#F8FAFC;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align:center;
            margin-bottom: 20px;
        ">
            <h3 style="color:#1E293B;">📤 Unggah Gambar Anda</h3>
            <p style="color:#64748B;">Pilih file dengan format .jpg, .jpeg, atau .png</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

        progress_text = "⏳ Sedang memproses gambar..."
        progress_bar = st.progress(0)
        st.text(progress_text)

        # ===============================
        # MODE: DETEKSI OBJEK (YOLO)
        # ===============================
        if mode == "Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()

            st.markdown("""
                <div style="
                    background-color:#FFFFFF;
                    border-left: 5px solid #2563EB;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    margin-top: 25px;
                ">
                    <h3 style="color:#1E3A8A;">🔍 Hasil Deteksi</h3>
                </div>
            """, unsafe_allow_html=True)

            st.image(result_img, use_container_width=True)

        # ===============================
        # MODE: KLASIFIKASI GAMBAR
        # ===============================
        elif mode == "Klasifikasi Gambar":
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)

            st.markdown("""
                <div style="
                    background-color:#FFFFFF;
                    border-left: 5px solid #16A34A;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    margin-top: 25px;
                ">
                    <h3 style="color:#166534;">🧩 Hasil Prediksi</h3>
                </div>
            """, unsafe_allow_html=True)

            st.write(f"**Kelas Prediksi:** {class_index}")
            st.write(f"**Probabilitas:** {float(np.max(prediction)):.2f}")

        progress_bar.progress(100)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===============================
    # Tombol Kembali
    # ===============================
    st.markdown("""
        <div style="text-align:center; margin-top:30px;">
    """, unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# SMALL FOOTER
# ==========================
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 AI Vision — dibuat oleh kamu</div>", unsafe_allow_html=True)
