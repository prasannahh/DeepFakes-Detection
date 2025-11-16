🧩 DeepFake Detection Using CNN & RNN

This project is a DeepFake Detection system that analyzes video frames to determine whether a video is real or fake. It combines MobileNetV2 (CNN) for spatial feature extraction and LSTM (RNN) for temporal pattern learning. The system includes a Streamlit web application where users can upload videos and get instant DeepFake detection results.

🚀 Features

* Upload any video file (MP4, AVI, MOV)
* Automatic frame extraction using OpenCV
* Preprocessing: resize → normalize → preprocess_input
* Deep learning pipeline:
* MobileNetV2 for feature extraction
* LSTM (RNN) for sequence analysis
* Frame-wise fake probability
* Final classification: Real or Fake
* Simple and interactive Streamlit UI

📈 Future Enhancements

* Live webcam DeepFake detection
* Face landmark-based analysis
* Using Vision Transformers (ViT)
* Improving accuracy with 3D CNN
* docker deployment
* Hosting on cloud or HuggingFace Spaces


