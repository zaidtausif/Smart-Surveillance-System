import cv2

COLORS = {
    "person": (40, 180, 40),
    "bag":    (40, 40, 220),
    "bicycle":(220, 180, 40),
    "alert":  (0, 0, 255)
}

def label_for(cls_id:int, names):
    try:
        return names[cls_id]
    except Exception:
        return str(cls_id)

def draw_tracks(frame, tracked, alerts, names):
    # boxes + labels
    for t in tracked:
        x1,y1,x2,y2 = map(int, t["xyxy"])
        lbl = t["label"]
        color = COLORS.get(lbl, (80,180,200))
        cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{lbl} #{t['id']}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # alerts banner
    for a in alerts:
        if a.get("xyxy"):
            x1,y1,x2,y2 = map(int, a["xyxy"])
            cv2.rectangle(frame,(x1,y1),(x2,y2), COLORS["alert"], 2)
            cv2.putText(frame, f"ALERT: {a['type']} {a['label']} #{a['id']}",
                        (x1, max(0,y1-22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["alert"], 2)
