import math
from collections import defaultdict, deque

class LoiteringRule:
    """
    Flags a PERSON who stays nearly stationary (low displacement) for a time window.
    """
    def __init__(self, fps, window_sec=12, min_disp_px=40):
        self.fps = max(1, int(fps))
        self.win = int(window_sec * self.fps)
        self.min_disp = float(min_disp_px)
        self.hist = defaultdict(lambda: deque(maxlen=self.win))
        self.last_alert_frame = {}

    @staticmethod
    def _centroid(xyxy):
        x1,y1,x2,y2 = xyxy
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update(self, tracked, frame_id, video_time_sec):
        alerts = []
        # update histories of persons only
        for t in tracked:
            if t["label"] != "person": 
                continue
            c = self._centroid(t["xyxy"])
            self.hist[t["id"]].append(c)

        # check displacement over window
        for t in tracked:
            if t["label"] != "person":
                continue
            tid = t["id"]
            H = self.hist[tid]
            if len(H) >= max(6, int(self.fps*0.5)):  # at least 0.5s
                x0,y0 = H[0]
                x1,y1 = H[-1]
                disp = math.hypot(x1-x0, y1-y0)
                if disp < self.min_disp and len(H) == H.maxlen:  # stationary for full window
                    # de-dup within ~3 seconds
                    if tid not in self.last_alert_frame or (frame_id - self.last_alert_frame[tid]) > int(self.fps*3):
                        alerts.append({
                            "type": "LOITERING",
                            "label": "person",
                            "id": int(tid),
                            "score": float(max(0.0, (self.min_disp - disp) / self.min_disp)),
                            "frame": int(frame_id),
                            "video_time_sec": float(video_time_sec),
                            "xyxy": list(map(int, t["xyxy"]))
                        })
                        self.last_alert_frame[tid] = frame_id
        return alerts
