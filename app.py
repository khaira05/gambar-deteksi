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
def load_models_cached():
    yolo = None
    clf = None
    yolo_err = None
    clf_err = None
    # try load YOLO
    try:
        # adjust path if your .pt is elsewhere
        if os.path.exists("model/best.pt"):
            yolo = YOLO("model/best.pt")
        else:
            yolo_err = "model/best.pt not found"
    except Exception as e:
        yolo_err = str(e)
    # try load classifier
    try:
        if os.path.exists("model/classifier_model.h5"):
            clf = tf.keras.models.load_model("model/classifier_model.h5")
        else:
            clf_err = "model/classifier_model.h5 not found"
    except Exception as e:
        clf_err = str(e)
    return yolo, clf, yolo_err, clf_err

yolo_model, classifier, yolo_err, clf_err = load_models_cached()

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

# ==========================
# CLASSIFICATION PAGE
# ==========================
elif st.session_state.page == "classify":
    # header with back button
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Beranda", key="back_from_class"):
        goto("home")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; margin-top:6px;'>Klasifikasi Gambar</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6f5f6f;'>Unggah gambar lalu klik tombol <b>Analisis</b></p>", unsafe_allow_html=True)

    # two-column layout: left upload, right result
    left, right = st.columns([1,1.1])

    with left:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Pilih file gambar (jpg/png/jpeg)", type=["jpg","jpeg","png"], key="upload_class")
        st.caption("Rekomendasi: ukuran gambar jelas & fokus pada objek")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        # Show placeholder before upload
        st.markdown("<div style='text-align:center; color:#6f5f6f;'>Hasil akan tampil di sini setelah analisis</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # when file uploaded, show preview + analyse button
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("Gagal membaca file gambar. Coba file lain.")
            img = None

        if img is not None:
            st.image(img, caption="Preview", use_column_width=True)

            # ANALYZE button
            if st.button("Analisis", key="analyse_class"):
                # handle classifier presence
                if classifier is None:
                    st.error("Model klasifikasi tidak tersedia. Detail: " + (clf_err or "unknown"))
                else:
                    with st.spinner("Melakukan klasifikasi..."):
                        # preprocessing
                        img_resized = img.resize((224,224))
                        arr = image.img_to_array(img_resized)
                        arr = np.expand_dims(arr, axis=0) / 255.0
                        preds = classifier.predict(arr)
                        idx = int(np.argmax(preds))
                        prob = float(np.max(preds))

                    # Map idx to label if you have mapping. Here we show index + prob.
                    # If you have a list `labels = ['cat','dog',...]` replace below:
                    labels = None
                    if labels:
                        label_text = labels[idx]
                    else:
                        label_text = f"Kelas {idx}"

                    # display results in right column area
                    st.markdown("<div class='result-box' style='margin-top:12px;'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align:center'>{label_text}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; color:#6f5f6f; margin-bottom:8px;'>Confidence: <b>{prob:.2%}</b></p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# DETECTION PAGE
# ==========================
elif st.session_state.page == "detect":
    # header with back button
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Beranda", key="back_from_detect"):
        goto("home")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; margin-top:6px;'>Deteksi Objek (YOLO)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6f5f6f;'>Unggah gambar lalu klik tombol <b>Deteksi</b></p>", unsafe_allow_html=True)

    left, right = st.columns([1,1.1])

    with left:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded2 = st.file_uploader("Pilih file gambar (jpg/png/jpeg)", type=["jpg","jpeg","png"], key="upload_detect")
        st.caption("Rekomendasi: objek terlihat jelas dalam foto")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; color:#6f5f6f;'>Hasil deteksi akan tampil di sini setelah proses</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded2:
        try:
            img2 = Image.open(uploaded2).convert("RGB")
        except Exception:
            st.error("Gagal membaca file gambar. Coba file lain.")
            img2 = None

        if img2 is not None:
            st.image(img2, caption="Preview", use_column_width=True)

            if st.button("Deteksi", key="do_detect"):
                if yolo_model is None:
                    st.error("Model YOLO tidak tersedia. Detail: " + (yolo_err or "unknown"))
                else:
                    with st.spinner("Mendeteksi objek..."):
                        # run yolo.ultralytics accepts PIL Image
                        results = yolo_model(img2)
                        # results[0].plot() returns numpy array (BGR in some versions) or PIL, cast safely
                        rendered = results[0].plot()
                        # if result is numpy array, convert to PIL for display
                        if isinstance(rendered, np.ndarray):
                            # if BGR -> convert to RGB (ultralytics sometimes returns RGB already)
                            try:
                                # detect channel order by checking max
                                rendered_rgb = rendered[..., ::-1]
                                disp_img = Image.fromarray(rendered_rgb)
                            except Exception:
                                disp_img = Image.fromarray(rendered)
                        elif isinstance(rendered, Image.Image):
                            disp_img = rendered
                        else:
                            # fallback: try to convert buffer
                            try:
                                disp_img = Image.fromarray(np.array(rendered))
                            except Exception:
                                disp_img = None

                    if disp_img:
                        st.image(disp_img, caption="Hasil Deteksi", use_column_width=True)
                    else:
                        st.error("Gagal menampilkan hasil deteksi (format output tidak dikenali).")

# ==========================
# SMALL FOOTER
# ==========================
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 AI Vision — dibuat oleh kamu</div>", unsafe_allow_html=True)
