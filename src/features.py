# src/features.py
import numpy as np
from collections import deque

class TrackBuffer:
    """
    Maintains a short history (deque) per track and provides feature vectors.
    """

    def __init__(self, max_frames=30):
        # store per-track deque of (cx, cy, w, h, frame_id)
        self.hist = {}
        self.max_frames = max_frames

    def update(self, tid, xyxy, frame_id):
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        if tid not in self.hist:
            self.hist[tid] = deque(maxlen=self.max_frames)
        self.hist[tid].append((cx, cy, w, h, frame_id))

    def get_history(self, tid):
        return list(self.hist.get(tid, []))

    def prune(self, current_frame, max_inactive_frames=150):
        # remove tracks not updated in a while
        to_delete = []
        for tid, dq in self.hist.items():
            if len(dq) == 0:
                to_delete.append(tid)
            elif (current_frame - dq[-1][4]) > max_inactive_frames:
                to_delete.append(tid)
        for tid in to_delete:
            del self.hist[tid]

def trajectory_features(history, fps=30):
    """
    Given history list of (cx,cy,w,h,frame_id) ordered oldest->newest,
    return a fixed-length feature vector.
    """
    if len(history) < 3:
        # not enough history: return a default small vector
        return np.zeros(12, dtype=np.float32)

    cxs = np.array([h[0] for h in history])
    cys = np.array([h[1] for h in history])
    ws = np.array([h[2] for h in history])
    hs = np.array([h[3] for h in history])
    frames = np.array([h[4] for h in history])

    # time deltas in seconds (assume near-constant frame rate if desired)
    if len(frames) >= 2:
        dt = (frames[-1] - frames[0]) / float(fps)
        if dt <= 0:
            dt = 1.0 / fps
    else:
        dt = 1.0 / fps

    # displacement and path length
    dx = cxs[-1] - cxs[0]
    dy = cys[-1] - cys[0]
    displacement = np.sqrt(dx*dx + dy*dy)
    diffs = np.sqrt(np.diff(cxs)**2 + np.diff(cys)**2)
    path_len = np.sum(diffs)

    # speeds
    mean_speed = path_len / dt
    max_step = diffs.max() if len(diffs) > 0 else 0.0
    speed_std = diffs.std() if len(diffs) > 0 else 0.0

    # direction variance (angle variance)
    angles = np.arctan2(np.diff(cys), np.diff(cxs) + 1e-6)
    ang_var = float(np.nanvar(angles)) if angles.size > 0 else 0.0

    # size stats
    mean_w = float(np.mean(ws))
    mean_h = float(np.mean(hs))
    size_var = float(np.var(ws) + np.var(hs))

    # dwell metric: how much it stayed in small bounding region
    bbox_minx = cxs.min()
    bbox_maxx = cxs.max()
    bbox_miny = cys.min()
    bbox_maxy = cys.max()
    span = max(bbox_maxx - bbox_minx, bbox_maxy - bbox_miny)
    dwell_frac = 1.0 - min(1.0, span / (max(1.0, max(mean_w, mean_h) * 10.0)))

    # build fixed vector: 12 dims
    feat = np.array([
        displacement,
        path_len,
        mean_speed,
        speed_std,
        max_step,
        ang_var,
        mean_w,
        mean_h,
        size_var,
        span,
        dwell_frac,
        len(history)
    ], dtype=np.float32)

    # normalize / clip to reasonable values later
    return feat
