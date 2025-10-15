import os, sys, json, time, sqlite3
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = r"D:\Kuliah\Semester 6\TPT\Belajar Opencv_5\video_pagi_kanan.mp4"
DB_PATH = r"D:\Kuliah\Semester 6\TPT\Belajar Opencv_5\counter.db"
TABLE = "vehicle_counts"

WEIGHTS = "yolov8m.pt"
CONF = 0.50
IOU  = 0.50
VEHICLE_CLASSES = {2, 3, 5, 7}
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def open_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT,
            frame_index INTEGER,
            t_video_sec REAL,
            vehicles_total INTEGER,
            class_counts TEXT,
            saved_at_utc TEXT DEFAULT (datetime('now'))
        );
    """)
    return conn, cur

def main():
    conn, cur = open_db()
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time = 1.0 / fps
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        res = model.predict(source=frame, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
        det = res.boxes
        count_now = 0
        per_class = {CLASS_NAMES[c]: 0 for c in VEHICLE_CLASSES}

        if det is not None:
            clss = det.cls.int().cpu().tolist() if det.cls is not None else []
            for c in clss:
                if c in VEHICLE_CLASSES:
                    count_now += 1
                    per_class[CLASS_NAMES[c]] += 1

        cur.execute("BEGIN")
        cur.execute(f"""
            INSERT INTO {TABLE}
            (video_path, frame_index, t_video_sec, vehicles_total, class_counts)
            VALUES (?, ?, ?, ?, ?)
        """, (VIDEO_PATH, frame_idx, frame_idx/fps, count_now, json.dumps(per_class)))
        cur.execute("COMMIT")

        print(f"[{frame_idx}] count={count_now} | {per_class}")
        time.sleep(frame_time)

if __name__ == "__main__":
    main()
