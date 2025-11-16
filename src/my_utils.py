import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_img(path, target_size=(224,224)):
    img = image.load_img(path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return arr

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def extract_frames_from_video(video_path, out_dir, every_n_frames=1, resize=(224,224), max_frames=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize)
            fname = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved