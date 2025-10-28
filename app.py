# ====================================================
# 1️⃣ IMPORT LIBRARY & KONFIGURASI HALAMAN
# ====================================================
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="AI Vision App", page_icon="🤖", layout="centered")

# ====================================================
# 2️⃣ LOAD MODEL (TIDAK DIUBAH)
# ====================================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

# Panggil fungsi load_models()
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
# 1️⃣ KONFIGURASI HALAMAN
# ====================================================
st.set_page_config(page_title="AI Vision", page_icon="🦩", layout="centered")

# ====================================================
# 2️⃣ INISIALISASI SESSION STATE UNTUK NAVIGASI
# ====================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "mode" not in st.session_state:
    st.session_state.mode = None  # deteksi atau klasifikasi

# ====================================================
# 3️⃣ CUSTOM CSS UNTUK TAMPILAN (GAYA AI VISION)
# ====================================================
st.markdown("""
    <style>
        body {
            background-color: #fff7fb;
        }
        .main {
            padding: 0rem 2rem;
        }
        .main-title {
            text-align:center;
            font-size:42px;
            font-weight:700;
            color:#212121;
            margin-top:20px;
        }
        .highlight {
            color:#d63384;
        }
        .subtitle {
            text-align:center;
            font-size:18px;
            color:#6B7280;
            margin-bottom:40px;
        }
        .menu-card {
            background-color:white;
            border-radius:20px;
            padding:25px;
            text-align:center;
            box-shadow:0 6px 15px rgba(0,0,0,0.08);
            transition:transform 0.2s;
        }
        .menu-card:hover {
            transform: scale(1.03);
        }
        .menu-card h4 {
            color:#d63384;
        }
        .btn-primary {
            display:inline-block;
            background: linear-gradient(90deg, #ff80ab, #d63384);
            color:white;
            padding:12px 32px;
            border-radius:30px;
            text-decoration:none;
            font-weight:600;
            text-align:center;
            margin-top:30px;
        }
        .btn-primary:hover {
            background: linear-gradient(90deg, #d63384, #ff80ab);
            text-decoration:none;
        }
        .metrics {
            text-align:center;
            font-size:1rem;
            color:#4b4b4b;
            margin-top:1.5rem;
        }
        .back-btn {
            text-align:center;
            margin-top:40px;
        }
        ul {
            text-align:left;
            color:#555;
            padding-left:1.2rem;
        }
    </style>
""", unsafe_allow_html=True)


# ====================================================
# 4️⃣ HALAMAN BERANDA (HOME)
# ====================================================
if st.session_state.page == "home":
    st.image("https://cdn.pixabay.com/photo/2018/09/19/19/42/flamingo-3690042_960_720.jpg", use_column_width=True)
    st.markdown('<h1 class="main-title">Selamat Datang di <span class="highlight">AI Vision</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Platform AI canggih untuk mendeteksi objek dan mengklasifikasikan jenis burung dalam gambar menggunakan teknologi deep learning terdepan.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="menu-card">
                <h4>🔍 Deteksi Objek</h4>
                <p>Identifikasi dan lokalisasi objek dalam gambar dengan presisi tinggi.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="menu-card">
                <h4>🖼️ Klasifikasi Gambar</h4>
                <p>Mengklasifikasi jenis burung dengan detail spesies dan karakteristik.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="menu-card">
                <h4>📈 Akurasi Tinggi</h4>
                <p>Hasil analisis dengan tingkat akurasi dan confidence score tinggi.</p>
            </div>
        """, unsafe_allow_html=True)

    # Tombol ke halaman berikut
    if st.button("🚀 Lanjut ke Pilih Mode Analisis", use_container_width=True):
        st.session_state.page = "analysis"

    # Metrics info
    st.markdown("""
    <div class="metrics">
        <b>95%</b> Akurasi &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>200+</b> Spesies Burung &nbsp;&nbsp; | &nbsp;&nbsp;
        <b><2s</b> Waktu Proses
    </div>
    """, unsafe_allow_html=True)


# ====================================================
# 5️⃣ HALAMAN PILIH MODE ANALISIS
# ====================================================
elif st.session_state.page == "analysis":
    st.markdown('<h1 class="main-title">Pilih Jenis Analisis <span class="highlight">AI Vision</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pilih mode analisis yang ingin Anda gunakan untuk menganalisis gambar burung.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="menu-card" style="border-top:5px solid #ff80ab;">
                <h4>🔎 Deteksi Objek</h4>
                <p>Temukan dan lokalisasi objek dalam gambar dengan bounding box.</p>
                <ul>
                    <li>Identifikasi posisi objek</li>
                    <li>Koordinat lokasi</li>
                    <li>Confidence score</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 Pilih Deteksi Objek", use_container_width=True):
            st.session_state.mode = "deteksi"
            st.session_state.page = "upload"

    with col2:
        st.markdown("""
            <div class="menu-card" style="border-top:5px solid #b45ef1;">
                <h4>📷 Klasifikasi Gambar</h4>
                <p>Identifikasi jenis dan spesies burung dengan detail lengkap.</p>
                <ul>
                    <li>Nama spesies</li>
                    <li>Karakteristik detail</li>
                    <li>Informasi habitat</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🦩 Pilih Klasifikasi", use_container_width=True):
            st.session_state.mode = "klasifikasi"
            st.session_state.page = "upload"

    # Tombol kembali
    if st.button("⬅️ Kembali ke Beranda", use_container_width=True):
        st.session_state.page = "home"


# ====================================================
# 6️⃣ HALAMAN INTI (UPLOAD GAMBAR)
# ====================================================
# ====================================================
# 🟦 HALAMAN UPLOAD GAMBAR (untuk deteksi / klasifikasi)
# ====================================================
elif st.session_state.page == "upload":

    # Judul halaman tergantung mode
    if st.session_state.mode == "deteksi":
        st.markdown('<h1 class="main-title">🔍 Deteksi Objek</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Unggah gambar untuk mendeteksi objek burung di dalamnya.</p>', unsafe_allow_html=True)
    elif st.session_state.mode == "klasifikasi":
        st.markdown('<h1 class="main-title">📷 Klasifikasi Gambar</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Unggah gambar burung untuk mengidentifikasi spesiesnya.</p>', unsafe_allow_html=True)

    # Upload file
    uploaded_image = st.file_uploader("📤 Unggah gambar di sini", type=["jpg", "jpeg", "png"])

    # Tombol kembali ke pilih mode
    if st.button("⬅️ Kembali ke Pilih Mode", use_container_width=True):
        st.session_state.page = "analysis"
        st.stop()  # hentikan eksekusi supaya tidak lanjut ke bawah

    # ====================================================
    # 🟦 MODE DETEKSI OBJEK (YOLO)
    # ====================================================
    if st.session_state.mode == "deteksi":
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            results = yolo_model(img)
            result_img = results[0].plot()

            st.markdown(f"""
                <div style="
                    background:white;
                    border-left: 6px solid #d63384;
                    border-radius:14px;
                    padding:22px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-top:25px;
                ">
                    <h3 style="color:#d63384; margin-bottom:10px;">🔍 Hasil Deteksi Objek</h3>
                    <p style="color:#334155; font-size:15px;">
                        Sistem AI Vision berhasil mendeteksi objek dalam gambar berikut menggunakan model YOLO.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.image(result_img, caption="🖼️ Hasil Deteksi", use_column_width=True)
        else:
            st.info("📸 Silakan unggah gambar terlebih dahulu untuk dideteksi.")

    # ====================================================
    # 🟩 MODE KLASIFIKASI GAMBAR
    # ====================================================
    elif st.session_state.mode == "klasifikasi":
        if uploaded_image is not None:
            # --- Preprocessing gambar ---
            img = Image.open(uploaded_image).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

            # --- Prediksi model ---
            with st.spinner("🔎 Sedang menganalisis gambar..."):
                prediction = classifier.predict(img_array)

            predicted_class = np.argmax(prediction, axis=1)[0]
            class_names = [
                "AMERICAN GOLDFINCH",
                "BARN OWL",
                "CARMINE BEE-EATER",
                "DOWNY WOODPECKER",
                "EMPEROR PENGUIN",
                "FLAMINGO"
            ]
            predicted_label = class_names[predicted_class]

            # --- Tampilkan hasil ---
            st.image(img_resized, caption="📸 Gambar yang Dianalisis", use_column_width=True)

            st.markdown(f"""
                <div style="
                    background:white;
                    border-left: 6px solid #10B981;
                    border-radius:14px;
                    padding:22px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-top:25px;
                ">
                    <h3 style="color:#10B981; margin-bottom:8px;">🧠 Hasil Klasifikasi Gambar</h3>
                    <p style="color:#334155; font-size:15px;">
                        Model AI Vision mengenali gambar sebagai spesies:
                        <b style="color:#10B981;">{predicted_label}</b>
                    </p>
                    <p style="color:#475569; margin-top:6px;">
                        Confidence Score: <b>{np.max(prediction):.2f}</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📸 Silakan unggah gambar terlebih dahulu untuk diklasifikasikan.")

# ====================================================
# 7️⃣ TOMBOL KEMBALI KE BERANDA
# ====================================================
    # Tombol kembali SELALU muncul, meskipun upload error atau stuck
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
    st.markdown('</div>', unsafe_allow_html=True)
