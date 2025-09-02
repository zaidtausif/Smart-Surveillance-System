# src/train_iso.py
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.tracker_utils import init_tracker, update_tracks
from src.features import TrackBuffer, trajectory_features
from src.anomaly_model import IsolationAnomaly

def iter_ucsd_sequence(seq_dir):
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.tif")))
    for f in frames:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # convert grayscale to RGB
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            yield img

def main():
    model = YOLO("yolov8n.pt")
    trackbuf = TrackBuffer(max_frames=30)
    feats = []

    base_dir = "data/UCSDped2/Train"
    seq_dirs = sorted(glob.glob(os.path.join(base_dir, "Train*")))

    fps = 30

    for seq in seq_dirs:
        print(f"Processing {seq} ...")
        frame_id = 0

        for frame in iter_ucsd_sequence(seq):
            frame_id += 1
            # **track() instead of model()**
            for result in model.track(source=frame, stream=True, tracker="trackers/bytetrack.yaml"):
                tracked = result.boxes  # now compatible with BYTETracker internally

                # update buffer + extract features
                if tracked is not None and len(tracked) > 0:
                    xyxy = tracked.xyxy.cpu().numpy()
                    ids = tracked.id.cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        tid = ids[i]
                        trackbuf.update(tid, xyxy[i], frame_id)
                        hist = trackbuf.get_history(tid)
                        if len(hist) >= 6:
                            feat = trajectory_features(hist, fps=fps)
                            feats.append(feat)

    feats = np.vstack(feats)
    print("Collected features shape:", feats.shape)

    iso = IsolationAnomaly()
    iso.train(feats, n_estimators=200, contamination=0.01)
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    main()
