# ====================================================
# 1️⃣ IMPORT LIBRARY & KONFIGURASI HALAMAN
# ====================================================
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Vision App", page_icon="🤖", layout="centered")

# ====================================================
# 2️⃣ LOAD MODEL (TIDAK DIUBAH)
# ====================================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

class_labels = [
    "AMERICAN GOLDFINCH",
    "BARN OWL",
    "CARMEN BEE-EATER",
    "DOWNY WOODPECKER",
    "EMPEROR PENGUIN",
    "FLAMINGO"
]

# ====================================================
# 3️⃣ INISIALISASI SESSION STATE UNTUK NAVIGASI
# ====================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ====================================================
# 4️⃣ CUSTOM CSS UNTUK TAMPILAN (ELEGAN & PROFESIONAL)
# ====================================================
st.markdown("""
    <style>
        body {
            background-color: #F1F5F9;
        }
        .main-title {
            text-align:center;
            font-size:42px;
            font-weight:700;
            color:#1E293B;
            margin-top:40px;
        }
        .subtitle {
            text-align:center;
            font-size:18px;
            color:#64748B;
            margin-bottom:40px;
        }
        .menu-card {
            background-color:white;
            border-radius:20px;
            padding:30px;
            text-align:center;
            box-shadow:0 6px 18px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .menu-card:hover {
            transform: scale(1.03);
        }
        .upload-card {
            background-color:#F8FAFC;
            border-radius:15px;
            padding:25px;
            text-align:center;
            box-shadow:0 4px 15px rgba(0,0,0,0.08);
            margin-bottom:25px;
        }
        .result-card {
            background-color:white;
            border-left:6px solid #2563EB;
            border-radius:12px;
            padding:20px 25px;
            box-shadow:0 4px 10px rgba(0,0,0,0.1);
            margin-top:25px;
        }
        .back-btn {
            text-align:center;
            margin-top:40px;
        }
    </style>
""", unsafe_allow_html=True)

# ====================================================
# 5️⃣ HALAMAN BERANDA (PILIH MODE)
# ====================================================
if st.session_state.page == "home":
    st.markdown('<h1 class="main-title">🤖 AI Vision App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deteksi dan Klasifikasi Gambar dengan Kecerdasan Buatan</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Deteksi Objek (YOLO)", use_container_width=True):
            st.session_state.page = "detect"

    with col2:
        if st.button("🧩 Klasifikasi Gambar", use_container_width=True):
            st.session_state.page = "classify"

# ====================================================
# 6️⃣ HALAMAN DETEKSI & KLASIFIKASI GAMBAR
# ====================================================
else:
    # Tentukan mode
    if st.session_state.page == "detect":
        mode = "Deteksi Objek (YOLO)"
        icon = "🎯"
        color = "#2563EB"
    else:
        mode = "Klasifikasi Gambar"
        icon = "🧩"
        color = "#16A34A"

    # Header tampilan
    st.markdown(f"""
        <div style="
            text-align:center;
            padding: 35px 20px 25px 20px;
            border-radius: 20px;
            background: linear-gradient(135deg, {color}, #1E293B);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            color: white;
            margin-bottom: 30px;
        ">
            <h1 style="font-size:38px; margin-bottom:5px;">{icon} {mode}</h1>
            <p style="font-size:16px; opacity:0.9;">Unggah gambar dan lihat hasil analisis AI</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload gambar
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Unggah Gambar")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Jika ada gambar diunggah
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Gambar yang diunggah", use_container_width=True)

        # Progress bar
        progress_text = "⏳ Sedang memproses gambar..."
        progress_bar = st.progress(0)
        st.text(progress_text)

        # ====================================================
        # 🟦 MODE DETEKSI OBJEK (YOLO)
        # ====================================================
        if mode == "Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()

            st.markdown(f"""
                <div class="result-card" style="border-left-color:{color};">
                    <h3 style="color:{color}; margin-bottom:10px;">🔍 Hasil Deteksi</h3>
                    <p style="color:#334155;">Berikut hasil deteksi objek dari gambar yang diunggah.</p>
                </div>
            """, unsafe_allow_html=True)

            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        # ====================================================
        # 🟩 MODE KLASIFIKASI GAMBAR
        # ====================================================
        elif mode == "Klasifikasi Gambar":
            uploaded_image = st.file_uploader("Upload gambar untuk klasifikasi", type=["jpg", "jpeg", "png"])

            if uploaded_image is not None:
                img = Image.open(uploaded_image)

            # Ambil ukuran input model otomatis
            input_shape = classifier.input_shape
            target_size = input_shape[1:3]
            channels = input_shape[3] if len(input_shape) > 3 else 3

            # Ubah format warna sesuai model
            if channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

            # Resize otomatis ke ukuran input model
            img_resized = img.resize(target_size)

            # Ubah ke array dan normalisasi
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]

            st.image(img_resized, caption="Gambar yang diproses", use_container_width=True)
            st.success(f"Hasil Prediksi: {predicted_class}")

            # Tampilkan hasil
            st.markdown(f"""
            <div class="result-card" style="border-left-color:{color};">
                <h3 style="color:{color}; margin-bottom:10px;">🧠 Hasil Prediksi</h3>
                <p style="color:#334155;">Model berhasil mengklasifikasikan gambar dengan hasil berikut:</p>
            </div>
        """, unsafe_allow_html=True)

st.write(f"**Kelas Prediksi:** {predicted_class}")
st.write(f"**Probabilitas:** {np.max(prediction):.2f}")
# ====================================================
# 7️⃣ TOMBOL KEMBALI KE BERANDA
# ====================================================
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
    st.markdown('</div>', unsafe_allow_html=True)
