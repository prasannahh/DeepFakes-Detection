import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from model_defs import build_video_model
from tensorflow.keras.callbacks import ModelCheckpoint
from evaluate import plot_training_history, evaluate_model

FRAME_DIR = r"D:\Prasanna\FF++"   # contains real/ and fake/ folders
TIMESTEPS = 16
BATCH = 4
IMG_SIZE = (224, 224)
EPOCHS = 12

class VideoSequence(Sequence):
    def __init__(self, frame_root, batch_size=BATCH, timesteps=TIMESTEPS, shuffle=True):
        self.samples = []
        for cls in ['real', 'fake']:
            cls_dir = os.path.join(frame_root, cls)
            if not os.path.exists(cls_dir):
                continue
            for vid in os.listdir(cls_dir):
                if vid.endswith('.mp4'):
                    self.samples.append((os.path.join(cls_dir, vid), 0 if cls == 'real' else 1))
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch), self.timesteps, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        y = np.zeros((len(batch),), dtype=np.float32)

        for i, (vid_path, label) in enumerate(batch):
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                continue

            indices = np.linspace(0, total_frames - 1, self.timesteps).astype(int)
            frames = []
            for frame_no in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, IMG_SIZE)
                frame = preprocess_input(frame)
                frames.append(frame)

            cap.release()
            while len(frames) < self.timesteps and len(frames) > 0:
                frames.append(frames[-1])
            if len(frames) == 0:
                continue

            X[i] = np.array(frames[:self.timesteps])
            y[i] = label

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

def main():
    seq = VideoSequence(FRAME_DIR, batch_size=BATCH, timesteps=TIMESTEPS)
    print(f"✅ Found {len(seq.samples)} video samples")

    model = build_video_model(frame_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), timesteps=TIMESTEPS, cnn_trainable=False)
    model.summary()

    checkpoint = ModelCheckpoint("video_model_best.h5", save_best_only=True, monitor="loss", mode="min")
    model.fit(seq, epochs=EPOCHS, callbacks=[checkpoint])
    model.save("video_model_final.h5")
    print("🎉 Training complete. Saved model as 'video_model_final.h5'")

if __name__ == "__main__":
    main()
    
# After your model.fit() call
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Plot accuracy/loss curves
plot_training_history(history)

# Evaluate model on test dataset
evaluate_model(model, test_ds)

