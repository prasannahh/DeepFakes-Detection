import os
from my_utils import extract_frames_from_video
from tqdm import tqdm

def process_videos(video_dir, out_root, every_n=5, max_frames=25):
    for cls in os.listdir(video_dir):
        cls_in = os.path.join(video_dir, cls)
        cls_out = os.path.join(out_root, cls)
        os.makedirs(cls_out, exist_ok=True)
        for fname in tqdm(os.listdir(cls_in)):
            if not fname.lower().endswith('.mp4'):
                continue
            fpath = os.path.join(cls_in, fname)
            out_folder = os.path.join(cls_out, os.path.splitext(fname)[0])
            os.makedirs(out_folder, exist_ok=True)
            extract_frames_from_video(fpath, out_folder, every_n_frames=every_n, resize=(224,224), max_frames=max_frames)

if __name__ == "__main__":
    process_videos("dataset/videos", "dataset/video_frames", every_n=3, max_frames=25)
