# src/utils/tracker_utils.py
import os
import yaml
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
import numpy as np

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

def init_tracker():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "trackers", "bytetrack.yaml")
    cfg_path = os.path.abspath(cfg_path)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"ByteTrack config not found at {cfg_path}")

    with open(cfg_path, "r") as f:
        args_dict = yaml.safe_load(f)
    args = dict_to_namespace(args_dict)

    return BYTETracker(args)

def update_tracks(tracker, detections):
    """Convert detections to BYTETracker expected format"""
    if not detections or len(detections) == 0:
        return []

    # detections: [[x1, y1, x2, y2, conf], ...]
    dets_for_tracker = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        dets_for_tracker.append(SimpleNamespace(xyxy=np.array([x1, y1, x2, y2]), conf=conf))

    return tracker.update(dets_for_tracker)
