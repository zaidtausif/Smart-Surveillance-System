import math
from collections import defaultdict, deque

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xi1, yi1 = max(ax1,bx1), max(ay1,by1)
    xi2, yi2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    aarea = max(0, ax2-ax1) * max(0, ay2-ay1)
    barea = max(0, bx2-bx1) * max(0, by2-by1)
    u = aarea + barea - inter
    return inter / u if u > 0 else 0.0

class AbandonmentRule:
    """
    Flags a BAG that remains stationary and unattended for N seconds.
    - stationary: bbox center displacement < bag_stationary_px in window
    - unattended: no PERSON centroid within 'near_px' for 'unattended_sec'
    """
    def __init__(self, fps, window_sec=6, bag_stationary_px=20, unattended_sec=10, near_px=120):
        self.fps = max(1, int(fps))
        self.win = int(window_sec * self.fps)
        self.unatt_frames = int(unattended_sec * self.fps)
        self.near_px = float(near_px)
        self.bag_hist = defaultdict(lambda: deque(maxlen=self.win))
        self.bag_last_near_person = defaultdict(int)
        self.bag_last_alert_frame = {}
        self.bag_label_set = set(["backpack","handbag","suitcase","bag"])  # harmonize

    @staticmethod
    def _centroid(xyxy):
        x1,y1,x2,y2 = xyxy
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update(self, tracked, frame_id, video_time_sec):
        alerts = []
        persons = [t for t in tracked if t["label"] == "person"]
        bags    = [t for t in tracked if t["label"] in self.bag_label_set]

        # update bag histories
        for b in bags:
            self.bag_hist[b["id"]].append(self._centroid(b["xyxy"]))

        # compute if bag is stationary
        for b in bags:
            tid = b["id"]
            H = self.bag_hist[tid]
            stationary = False
            if len(H) >= max(6, int(self.fps*0.5)):
                x0,y0 = H[0]
                x1,y1 = H[-1]
                disp = math.hypot(x1-x0, y1-y0)
                stationary = disp < 20.0  # override if you want 'bag_stationary_px'

            # compute nearest person distance
            bc = self._centroid(b["xyxy"])
            near_any = False
            min_d = 1e9
            for p in persons:
                pc = self._centroid(p["xyxy"])
                d = math.hypot(pc[0]-bc[0], pc[1]-bc[1])
                if d < min_d:
                    min_d = d
                if d <= self.near_px:
                    near_any = True
                    break

            # Update last near timestamp
            if near_any:
                self.bag_last_near_person[tid] = frame_id

            # Abandonment if stationary + no near person for long
            last_near = self.bag_last_near_person.get(tid, 0)
            unattended_long = (frame_id - last_near) >= self.unatt_frames

            if stationary and unattended_long:
                # de-dup at ~5s
                if tid not in self.bag_last_alert_frame or (frame_id - self.bag_last_alert_frame[tid]) > int(self.fps*5):
                    alerts.append({
                        "type": "ABANDONED_BAG",
                        "label": "bag",
                        "id": int(tid),
                        "score": 1.0,
                        "frame": int(frame_id),
                        "video_time_sec": float(video_time_sec),
                        "xyxy": list(map(int, b["xyxy"])),
                        "extra": f"min_person_dist={min_d:.1f}px"
                    })
                    self.bag_last_alert_frame[tid] = frame_id

        return alerts
