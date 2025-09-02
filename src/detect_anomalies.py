import os, cv2, time, argparse, glob, numpy as np
from ultralytics import YOLO

from src.rules.loitering import LoiteringRule
from src.rules.abandonment import AbandonmentRule
from src.utils.draw import draw_tracks, label_for
from src.utils.logger import log_alert

# Define tracked object categories
WANTED_LABELS = {
    "person": {"ids": [0]},                 # coco: person
    "bicycle": {"ids": [1]},                # coco: bicycle
    "bag": {"ids": [24, 26, 28]},           # backpack(24), handbag(26), suitcase(28)
}

def get_tracker_cfg():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.abspath(os.path.join(here, "..", "trackers", "strongsort.yaml"))
    return cfg if os.path.exists(cfg) else "strongsort.yaml"

def map_label(cls_id: int, names):
    """Map YOLO class id to our tracked categories."""
    raw = label_for(cls_id, names)
    raw_lower = raw.lower()
    if raw_lower in ["backpack", "handbag", "suitcase", "bag"]:
        return "bag"
    if raw_lower == "bicycle":
        return "bicycle"
    if raw_lower == "person":
        return "person"
    for k, v in WANTED_LABELS.items():
        if cls_id in v["ids"]:
            return k
    return None

def iter_tif_sequence(seq_dir):
    """Yield frames from .tif sequences (UCSD/Avenue)."""
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.tif")))
    for f in frames:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            yield img

def process_video(model, tracker_cfg, video_path, parent_folder, args):
    """Process a single .avi/.mp4/.mov video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    names = model.model.names if hasattr(model.model, "names") else {}

    loiter_rule = LoiteringRule(fps=fps, window_sec=12, min_disp_px=40)
    abandon_rule = AbandonmentRule(fps=fps, window_sec=6, bag_stationary_px=20,
                                   unattended_sec=12, near_px=140)

    writer = None
    if args.save:
        os.makedirs("outputs/videos", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join("outputs", "videos",
                                f"out_{parent_folder}_{os.path.basename(video_path).rsplit('.',1)[0]}_{int(time.time())}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print("Saving to:", out_path)

    frame_id = 0
    for res in model.track(source=video_path, stream=True, conf=args.conf,
                           tracker=tracker_cfg, persist=True, imgsz=960, iou=0.5):
        frame = res.orig_img
        frame_id += 1
        video_time_sec = frame_id / fps

        tracked = []
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            ids = res.boxes.id.cpu().numpy().astype(int) if getattr(res.boxes, "id", None) is not None else np.arange(len(xyxy))

            for i in range(len(xyxy)):
                lbl = map_label(int(clss[i]), names)
                if lbl is None:
                    continue
                x1, y1, x2, y2 = xyxy[i].tolist()
                tracked.append({"id": int(ids[i]), "xyxy": [x1, y1, x2, y2], "label": lbl})

        alerts = []
        if tracked:
            alerts += loiter_rule.update(tracked, frame_id, video_time_sec)
            alerts += abandon_rule.update(tracked, frame_id, video_time_sec)

        draw_tracks(frame, tracked, alerts, names)

        for a in alerts:
            a["source_video"] = os.path.basename(video_path)
            a["source_folder"] = parent_folder
            log_alert(a, frame)

        if writer:
            writer.write(frame)

        if args.show:
            cv2.imshow("Surveillance (YOLOv8 + StrongSORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

def process_tif_folder(model, tracker_cfg, main_folder, args):
    """Process a folder with multiple TestXXX .tif sequences."""
    seq_dirs = sorted(glob.glob(os.path.join(main_folder, "Test*")))
    for seq in seq_dirs:
        print(f"Processing sequence: {seq}")
        frames = list(iter_tif_sequence(seq))
        if not frames:
            continue

        fps = 30
        height, width = frames[0].shape[:2]
        names = model.model.names if hasattr(model.model, "names") else {}

        loiter_rule = LoiteringRule(fps=fps, window_sec=12, min_disp_px=40)
        abandon_rule = AbandonmentRule(fps=fps, window_sec=6, bag_stationary_px=20,
                                       unattended_sec=12, near_px=140)

        writer = None
        if args.save:
            os.makedirs("outputs/videos", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join("outputs", "videos",
                                    f"out_{os.path.basename(main_folder)}_{os.path.basename(seq)}_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            print("Saving to:", out_path)

        for frame_id, frame in enumerate(frames, start=1):
            results = model.track(frame, stream=False, conf=args.conf,
                                  tracker=tracker_cfg, persist=True, imgsz=960, iou=0.5)
            for res in results:
                video_time_sec = frame_id / fps

                tracked = []
                if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                    ids = res.boxes.id.cpu().numpy().astype(int) if getattr(res.boxes, "id", None) is not None else np.arange(len(xyxy))

                    for i in range(len(xyxy)):
                        lbl = map_label(int(clss[i]), names)
                        if lbl is None:
                            continue
                        x1, y1, x2, y2 = xyxy[i].tolist()
                        tracked.append({"id": int(ids[i]), "xyxy": [x1, y1, x2, y2], "label": lbl})

                alerts = []
                if tracked:
                    alerts += loiter_rule.update(tracked, frame_id, video_time_sec)
                    alerts += abandon_rule.update(tracked, frame_id, video_time_sec)

                draw_tracks(frame, tracked, alerts, names)

                for a in alerts:
                    a["source_video"] = os.path.basename(seq)
                    a["source_folder"] = os.path.basename(main_folder)
                    log_alert(a, frame)

                if writer:
                    writer.write(frame)

                if args.show:
                    cv2.imshow("Surveillance (YOLOv8 + StrongSORT)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Single video file (.avi, .mp4, .mov)")
    ap.add_argument("--folder", help="Folder with .avi videos or TestXXX .tif folders")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    tracker_cfg = get_tracker_cfg()

    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(args.video)
        process_video(model, tracker_cfg, args.video, "single_video", args)

    elif args.folder:
        if not os.path.isdir(args.folder):
            raise FileNotFoundError(args.folder)
        avi_files = glob.glob(os.path.join(args.folder, "*.avi"))
        tif_folders = glob.glob(os.path.join(args.folder, "Test*"))
        if avi_files:
            for vid in avi_files:
                process_video(model, tracker_cfg, vid, os.path.basename(args.folder), args)
        elif tif_folders:
            process_tif_folder(model, tracker_cfg, args.folder, args)
        else:
            raise ValueError("No .avi videos or TestXXX folders found in input folder!")
    else:
        raise ValueError("You must provide either --video or --folder")

if __name__ == "__main__":
    main()
