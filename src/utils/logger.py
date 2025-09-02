import os, cv2, csv, time
from datetime import datetime

ALERT_DIR = os.path.join("outputs", "alerts")
SNAP_DIR  = os.path.join("outputs", "snaps")
LOG_PATH  = os.path.join(ALERT_DIR, "log.csv")

os.makedirs(ALERT_DIR, exist_ok=True)
os.makedirs(SNAP_DIR,  exist_ok=True)

# Ensure CSV header includes source info
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "video_time_sec", "type", "object_label", "track_id",
            "score", "frame", "snap_path", "source_video", "source_folder", "extra"
        ])

def log_alert(alert: dict, frame_bgr):
    """
    alert fields expected:
       type, label, id, score, frame, video_time_sec, xyxy, source_video, source_folder
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = f"{alert.get('type')}_{alert.get('label')}_{alert.get('id')}_{int(time.time()*1000)}.jpg"
    snap_path = os.path.join(SNAP_DIR, fname)

    # draw bounding box snapshot
    if frame_bgr is not None and alert.get("xyxy") is not None:
        x1, y1, x2, y2 = list(map(int, alert["xyxy"]))
        snap = frame_bgr.copy()
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(snap, f"{alert['type']} {alert['label']}#{alert['id']}",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(snap_path, snap)
    else:
        snap_path = ""

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            ts,
            f"{alert.get('video_time_sec', 0):.2f}",
            alert.get("type", ""),
            alert.get("label", ""),
            alert.get("id", ""),
            f"{alert.get('score', 0):.3f}",
            alert.get("frame", ""),
            snap_path,
            alert.get("source_video", ""),   # <-- NEW
            alert.get("source_folder", ""),  # <-- NEW
            alert.get("extra", "")
        ])
