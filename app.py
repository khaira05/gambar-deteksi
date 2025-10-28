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
    if st.button("🚀 Mulai Sekarang", use_container_width=True):
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
# 5️⃣ HALAMAN PILIH ANALISIS
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
        if st.button("🧠 Jalankan Deteksi Objek", use_container_width=True):
            st.success("Mode deteksi diaktifkan (placeholder).")

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
        if st.button("🦩 Jalankan Klasifikasi", use_container_width=True):
            st.success("Mode klasifikasi diaktifkan (placeholder).")

    # Tombol kembali
    st.markdown("<div class='back-btn'><a href='#' class='btn-primary' onclick='window.location.reload()'>← Kembali</a></div>", unsafe_allow_html=True)

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

            # 🔹 Pakai ukuran tetap biar cepat (gak perlu baca input_shape tiap kali)
            target_size = (224, 224)

            # Ubah format warna ke RGB
            img = img.convert("RGB")

            # 🔹 Resize gambar agar sesuai model
            img_resized = img.resize(target_size)

            # 🔹 Ubah ke array dan normalisasi
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # 🔹 Prediksi (pakai spinner biar ada indikator loading)
            with st.spinner("⏳ Sedang memproses gambar..."):
                prediction = classifier.predict(img_array)

            # Ambil kelas prediksi dan nama kelas
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

            # 🔹 Tampilkan hasil
            st.image(img_resized, caption="Gambar yang diproses", use_container_width=True)
            st.success(f"Hasil Prediksi: {predicted_label}")

            st.markdown(f"""
                <div class="result-card" style="border-left-color:#10b981;">
                    <h3 style="color:#10b981; margin-bottom:10px;">🧠 Hasil Prediksi</h3>
                    <p style="color:#334155;">Model berhasil mengklasifikasikan gambar dengan hasil berikut:</p>
                </div>
            """, unsafe_allow_html=True)

            st.write(f"**Kelas Prediksi:** {predicted_label}")
            st.write(f"**Probabilitas:** {np.max(prediction):.2f}")

# ====================================================
# 7️⃣ TOMBOL KEMBALI KE BERANDA
# ====================================================
    # Tombol kembali SELALU muncul, meskipun upload error atau stuck
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
    st.markdown('</div>', unsafe_allow_html=True)
