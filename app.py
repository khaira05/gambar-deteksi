import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ============================================================
# 📸 Aplikasi Deteksi & Klasifikasi Gambar (Tampilan Modern)
# ============================================================

# ==========================
# 1️⃣ Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="Deteksi & Klasifikasi Gambar", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #f8fafc;
        }
        h1, h2, h3, h4 {
            color: #1f2937;
        }
        .stProgress > div > div > div > div {
            background-color: #10b981;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 2️⃣ Sidebar Navigasi
# ==========================
st.sidebar.title("🧭 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["🏠 Beranda", "🎯 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar"]
)

# ==========================
# 3️⃣ Load Model
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
        return yolo_model, classifier
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

yolo_model, classifier = load_models()

# ==========================
# 4️⃣ Halaman Utama
# ==========================
if menu == "🏠 Beranda":
    st.title("📸 Aplikasi Deteksi & Klasifikasi Gambar")
    st.write("Unggah gambar dan biarkan AI menganalisisnya secara otomatis!")
    st.markdown("""
    #### Fitur yang Tersedia:
    - **🎯 Deteksi Objek (YOLO):** Mengenali berbagai objek dalam gambar secara otomatis.  
    - **🧠 Klasifikasi Gambar:** Mengidentifikasi kategori utama dari gambar yang diunggah.  
    """)
    st.image("assets/ai_banner.jpg", use_container_width=True)

# ==========================
# 5️⃣ Halaman Deteksi & Klasifikasi
# ==========================
elif menu in ["🎯 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar"]:
    st.title(menu)
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        with col2:
            progress_bar = st.progress(0)
            st.text("⏳ Sedang memproses gambar...")

            # ==========================
            # DETEKSI OBJEK (YOLO)
            # ==========================
            if menu == "🎯 Deteksi Objek (YOLO)":
                if yolo_model:
                    results = yolo_model(img)
                    result_img = results[0].plot()

                    progress_bar.progress(100)
                    st.success("✅ Deteksi selesai!")

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(result_img, caption="🔍 Hasil Deteksi", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Model YOLO belum dimuat atau file model tidak ditemukan.")

            # ==========================
            # KLASIFIKASI GAMBAR
            # ==========================
            elif menu == "🧠 Klasifikasi Gambar":
                if classifier:
                    img_resized = img.resize((224, 224))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    prediction = classifier.predict(img_array)
                    class_index = np.argmax(prediction)
                    prob = np.max(prediction)

                    progress_bar.progress(100)
                    st.success("✅ Klasifikasi selesai!")

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("### 🏷️ Hasil Prediksi:")
                    st.write(f"**Kelas:** {class_index}")
                    st.metric("Probabilitas", f"{prob:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Model klasifikasi belum dimuat atau file model tidak ditemukan.")
