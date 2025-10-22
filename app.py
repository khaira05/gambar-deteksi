import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

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
# Custom Tema 
# ==========================
st.markdown("""
<style>
    /* 🌸 Tampilan utama aplikasi */
    .stApp {
        background-color: #f5e6f1;  /* Warna lembut untuk background utama */
        font-family: 'Segoe UI', sans-serif;
    }

    /* 🎨 Warna dan gaya judul */
    h1, h2, h3 {
        color: #4B4453;  
    }

    /* 🪞 Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f5e6f1 !important;  /* Warna sidebar */
        color: #4B4453;
        border-right: 2px solid #e1cbe6; /* garis pembatas halus */
    }

    [data-testid="stSidebar"] * {
        color: #4B4453 !important;  /* Warna teks di sidebar */
    }

    /* ✨ Kotak sidebar */
    .css-1v3fvcr, .css-1d391kg {
        border-radius: 15px;
        padding: 10px;
    }

    /* 🪶 Hover effect di sidebar */
    [data-testid="stSidebar"] .css-1v3fvcr:hover {
        background-color: rgba(155, 89, 182, 0.15); /* ungu transparan */
        border-radius: 10px;
    }

    /* 🎯 Tombol Streamlit (misal tombol upload atau action) */
    div.stButton > button:first-child {
        background-color: #ba68c8;
        color: white;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }

    div.stButton > button:first-child:hover {
        background-color: #ab47bc;
        color: white;
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("📂 Pilih Mode:", 
                            ["🏠 Beranda", "Deteksi Objek (YOLO)", "Klasifikasi Gambar", "📘 Cara Menggunakan", "ℹ️ Tentang"])

# ==========================
# 1️⃣ Beranda
# ==========================
if menu == "🏠 Beranda":
    st.markdown("### Selamat datang di aplikasi deteksi & klasifikasi gambar 👋")
    st.write("Gunakan sidebar untuk memilih mode yang kamu mau.")

# ==========================
# 2️⃣ Deteksi & Klasifikasi
# ==========================
elif menu in ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]:
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Progress bar placeholder
        progress_text = "⏳ Sedang memproses gambar..."
        progress_bar = st.progress(0)
        st.text(progress_text)

        if menu == "Deteksi Objek (YOLO)":
            # Deteksi objek
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        elif menu == "Klasifikasi Gambar":
            # Preprocessing
            img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            st.write("### Hasil Prediksi:", class_index)
            st.write("Probabilitas:", np.max(prediction))

# ==========================
# 3️⃣ CARA MENGGUNAKAN (Bagian 4)
# ==========================
elif menu == "📘 Cara Menggunakan":
    st.markdown("""
    ## 📘 Panduan Penggunaan Aplikasi
    1. Pilih mode yang kamu inginkan dari sidebar (Deteksi atau Klasifikasi).  
    2. Unggah gambar yang ingin dianalisis (format .jpg, .jpeg, atau .png).  
    3. Tunggu proses analisis selesai.  
    4. Lihat hasil prediksi dan probabilitas di layar.  
    5. Gunakan hasil deteksi atau klasifikasi sesuai kebutuhanmu.  

    💡 **Tips:** Gunakan gambar yang jelas dan fokus agar hasil lebih akurat.
    """)

# ==========================
# 4️⃣ TENTANG
# ==========================
elif menu == "ℹ️ Tentang":
    st.markdown("""
    ### Tentang Aplikasi
    Aplikasi ini menggabungkan dua model AI:
    - **YOLOv8** untuk deteksi objek
    - **CNN (TensorFlow)** untuk klasifikasi gambar  
    Dibuat oleh: *Khaira Putri Syalaisa* 🎓
    """)
