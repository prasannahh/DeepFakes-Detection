'''
import streamlit as st
import numpy as np
from PIL import Image
import cv2, os, tempfile
import tensorflow as tf
from my_utils import load_img, extract_frames_from_video
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="DeepFake Detector", layout="centered")
st.title("DeepFake Detector (Image & Video)")
st.write("Upload an image or video to test your trained models.")

@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("image_model.h5")
    vid_model = tf.keras.models.load_model("video_model.h5")
    return img_model, vid_model

if not os.path.exists("image_model.h5") or not os.path.exists("video_model.h5"):
    st.warning("❗ Models not found. Train them first using train_image.py and train_video.py.")
else:
    img_model, vid_model = load_models()

    uploaded = st.file_uploader("Upload image or video (jpg/png/mp4)", type=['jpg','jpeg','png','mp4'])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        fname = tfile.name
        ext = os.path.splitext(uploaded.name)[1].lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)
            x = load_img(fname, (224,224))
            x = np.expand_dims(x, 0)
            pred = img_model.predict(x)[0,0]
            st.write(f"Probability Fake: {pred:.3f}")
            st.progress(int(pred*100))
            if pred > 0.5:
                st.error("Prediction: FAKE")
            else:
                st.success("Prediction: REAL")

        elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
            st.video(fname)
            tmpdir = tempfile.mkdtemp()
            extract_frames_from_video(fname, tmpdir, every_n_frames=3, resize=(224,224), max_frames=16)
            frames = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.jpg')])

            if len(frames) < 4:
                st.error("Not enough frames for analysis.")
            else:
                X = np.zeros((1, 16, 224, 224, 3), dtype=np.float32)
                idxs = np.linspace(0, len(frames)-1, 16).astype(int)
                for i, ix in enumerate(idxs):
                    img = load_img(frames[ix], (224,224))
                    X[0,i] = img
                pred = vid_model.predict(X)[0,0]
                st.write(f"Probability Fake: {pred:.3f}")
                st.progress(int(pred*100))
                if pred > 0.5:
                    st.error("Prediction: FAKE")
                else:
                    st.success("Prediction: REAL")
'''
import streamlit as st
import numpy as np
from PIL import Image
import cv2, os, tempfile
import tensorflow as tf
from my_utils import load_img, extract_frames_from_video
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 🎨 --- PAGE CONFIG ---
st.set_page_config(page_title="DeepFake Detector", layout="centered", page_icon="🎭")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f0f0f0;
}
h1 {
    color: #FFB400;
    text-align: center;
    font-family: 'Poppins', sans-serif;
}
.stProgress > div > div > div > div {
    background-color: #FF4B2B;
}
.uploadedFile {
    border-radius: 10px;
}
.result-card {
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
    text-align: center;
    background: rgba(255,255,255,0.1);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.result-text {
    font-size: 1.3rem;
    font-weight: bold;
}
.success-text {
    color: #00FFAB;
}
.error-text {
    color: #FF4B2B;
}
.info-text {
    color: #FFD700;
}
</style>
""", unsafe_allow_html=True)

# --- TITLE & HEADER ---
st.title("🎭 DeepFake Detector")
st.markdown("#### Upload an image or video to verify its authenticity.")
st.write("This tool uses AI models trained on real vs. fake samples to classify DeepFake content.")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("image_model.h5")
    vid_model = tf.keras.models.load_model("video_model.h5")
    return img_model, vid_model

if not os.path.exists("image_model.h5") or not os.path.exists("video_model.h5"):
    st.warning("⚠️ Models not found. Please train them using `train_image.py` and `train_video.py` first.")
else:
    img_model, vid_model = load_models()

    uploaded = st.file_uploader("📤 Upload image or video", type=['jpg','jpeg','png','mp4'])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        fname = tfile.name
        ext = os.path.splitext(uploaded.name)[1].lower()

        st.markdown("<hr>", unsafe_allow_html=True)

        if ext in ['.jpg', '.jpeg', '.png']:
            st.image(uploaded, caption="🖼️ Uploaded Image", use_container_width=True)
            x = load_img(fname, (224,224))
            x = np.expand_dims(x, 0)
            pred = img_model.predict(x)[0,0]

            st.markdown(f"<div class='result-card'><div class='info-text'>🔍 Analyzing image...</div></div>", unsafe_allow_html=True)
            st.progress(int(pred*100))
            
            if pred > 0.5:
                st.markdown(f"<div class='result-card'><div class='error-text'>❌ Prediction: FAKE<br>Probability: {pred:.3f}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card'><div class='success-text'>✅ Prediction: REAL<br>Probability: {1 - pred:.3f}</div></div>", unsafe_allow_html=True)

        elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
            st.video(fname)
            tmpdir = tempfile.mkdtemp()
            extract_frames_from_video(fname, tmpdir, every_n_frames=3, resize=(224,224), max_frames=16)
            frames = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.jpg')])

            if len(frames) < 4:
                st.error("⚠️ Not enough frames for analysis.")
            else:
                X = np.zeros((1, 16, 224, 224, 3), dtype=np.float32)
                idxs = np.linspace(0, len(frames)-1, 16).astype(int)
                for i, ix in enumerate(idxs):
                    img = load_img(frames[ix], (224,224))
                    X[0,i] = img
                pred = vid_model.predict(X)[0,0]
                st.markdown(f"<div class='result-card'><div class='info-text'>🎞️ Processing video frames...</div></div>", unsafe_allow_html=True)
                st.progress(int(pred*100))

                if pred > 0.5:
                    st.markdown(f"<div class='result-card'><div class='error-text'>🎭 Prediction: FAKE<br>Probability: {pred:.3f}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-card'><div class='success-text'>🧠 Prediction: REAL<br>Probability: {1 - pred:.3f}</div></div>", unsafe_allow_html=True)
