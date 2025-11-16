🧩 DeepFake Detection Using CNN & RNN

This project is a DeepFake Detection system that analyzes video frames to determine whether a video is real or fake. It combines MobileNetV2 (CNN) for spatial feature extraction and LSTM (RNN) for temporal pattern learning. The system includes a Streamlit web application where users can upload videos and get instant DeepFake detection results.

🚀 Features

Upload any video file (MP4, AVI, MOV)

Automatic frame extraction using OpenCV

Preprocessing: resize → normalize → preprocess_input

Deep learning pipeline:

MobileNetV2 for feature extraction

LSTM (RNN) for sequence analysis

Frame-wise fake probability

Final classification: Real or Fake

Simple and interactive Streamlit UI

🧠 Technology Stack
Component	Technology
Frontend	Streamlit
Backend	Python
Model	MobileNetV2 + LSTM
Video Processing	OpenCV
Libraries	TensorFlow, NumPy, PIL
📂 Workflow

User uploads a video through Streamlit

Frames are extracted using OpenCV

Each frame is preprocessed

MobileNetV2 extracts spatial features

LSTM analyzes frame sequences

Model outputs DeepFake probability

Final label: Real / Fake

📦 Dataset

This project is compatible with datasets such as:

FaceForensics++

Celeb-DF v2

Kaggle DeepFake Detection Dataset

(You can update this section based on your actual dataset.)

🛠️ Installation
Clone the Repository
git clone https://github.com/yourusername/your-reponame.git
cd your-reponame

Install Requirements
pip install -r requirements.txt

Run the Web App
streamlit run main.py

🖥️ Usage

Start the Streamlit app

Upload your video

Wait for frame extraction and model prediction

View the result (Real or Fake)

📈 Future Enhancements

Live webcam DeepFake detection

Face landmark-based analysis

Using Vision Transformers (ViT)

Improving accuracy with 3D CNN

Docker deployment

Hosting on cloud or HuggingFace Spaces

🔚 Conclusion

This DeepFake Detection system demonstrates how combining CNNs and RNNs can effectively capture both spatial and temporal patterns in videos to identify DeepFakes. The Streamlit interface makes it easy for users to test videos and understand the results.
