# parking_yolo_slots.py
import cv2, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ---------------- PATHS ----------------
VIDEO_PATH  = "/home/skyrunner/Documents/testing/video_new.mp4"
SLOTS_JSON  = "/home/skyrunner/Documents/testing/slots2.json"
OUT_DIR     = Path("/home/skyrunner/Documents/testing/results")
OUT_VIDEO   = OUT_DIR / "annotated_yolo.mp4"
OUT_CSV     = OUT_DIR / "occupancy_yolo.csv"

# Model YOLOv8 pretrained COCO (car=class id 2)
YOLO_WEIGHTS = "yolov8s.pt"   # boleh ganti: yolov8n.pt untuk lebih cepat

# -------------- PARAMS -----------------
CONF_THR = 0.25       # conf minimal deteksi
NMS_IOU  = 0.50       # iou NMS (bukan IoU poligon)
OVERLAP_THR = 0.08    # ambang overlap deteksi-slot (IoU mask) → tandai occupied
SHOW = True

# --------------------------------------
def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_slots(json_path, video_shape_hw):
    """Load polygon slot dan otomatis scale jika image_size di JSON beda dgn video."""
    H, W = video_shape_hw
    data = json.load(open(json_path, "r"))
    slots_raw = data.get("slots", [])
    meta_sz = data.get("meta", {}).get("image_size", [H, W])  # [H,W]
    H0, W0 = int(meta_sz[0]), int(meta_sz[1])
    sx, sy = W / max(1, W0), H / max(1, H0)

    slots = []
    for s in slots_raw:
        pts0 = np.array(s["points"], dtype=np.float32)
        pts = np.stack([pts0[:,0]*sx, pts0[:,1]*sy], axis=1).astype(np.int32)
        sid = int(s.get("id", len(slots)+1))
        cx, cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
        slots.append({"id": sid, "points": pts, "center": (cx, cy)})
    return slots

def polygon_mask(shape_hw, pts):
    h, w = shape_hw
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def bbox_mask(shape_hw, xyxy):
    h, w = shape_hw
    x1,y1,x2,y2 = map(int, xyxy)
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    m = np.zeros((h,w), dtype=np.uint8)
    if x2>x1 and y2>y1:
        cv2.rectangle(m, (x1,y1), (x2,y2), 255, -1)
    return m

def iou_masks(m1, m2):
    inter = np.logical_and(m1>0, m2>0).sum()
    if inter == 0: return 0.0
    union = np.logical_or(m1>0, m2>0).sum()
    return inter/union if union else 0.0

def main():
    ensure_outdir(OUT_DIR)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Gagal membuka video: {VIDEO_PATH}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # siapkan writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, FPS, (W, H))

    # load slots (autoscale)
    slots = load_slots(SLOTS_JSON, (H, W))
    if not slots:
        raise SystemExit("slots.json kosong—annotasi dulu slot parkirnya.")

    # precompute mask poligon slot
    slot_masks = [polygon_mask((H,W), s["points"]) for s in slots]

    # load YOLO
    model = YOLO(YOLO_WEIGHTS)
    # class 'car' di COCO adalah id 2; pakai filter di inferensi
    car_class_id = 2

    rows = []
    frame_idx = 0

    if SHOW:
        cv2.namedWindow("Parking YOLOv8", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking YOLOv8", min(W,1280), min(H,720))

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        # YOLO inference (satu frame)
        res = model.predict(
            source=frame, conf=CONF_THR, iou=NMS_IOU, verbose=False, classes=[car_class_id]
        )[0]

        # ambil deteksi mobil → list bbox
        det_bboxes = []
        if res.boxes is not None and len(res.boxes)>0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            for box in xyxy:
                x1,y1,x2,y2 = box.astype(int)
                det_bboxes.append((x1,y1,x2,y2))
                # gambar bbox tipis
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

        # assign ke slot via IoU mask
        slot_taken = [False]*len(slots)
        # precompute mask bbox
        det_masks = [bbox_mask((H,W), b) for b in det_bboxes]

        for s_idx, sm in enumerate(slot_masks):
            # cari overlap terbaik dengan deteksi manapun
            best = 0.0
            for dm in det_masks:
                iou = iou_masks(dm, sm)
                if iou > best: best = iou
            if best >= OVERLAP_THR:
                slot_taken[s_idx] = True

        occupied = int(sum(slot_taken))

        # gambar poligon slot
        for i, s in enumerate(slots):
            color = (0,0,255) if slot_taken[i] else (0,200,0)  # merah=isi, hijau=kosong
            cv2.polylines(frame, [s["points"]], True, color, 2)
            cx, cy = s["center"]
            cv2.putText(frame, str(s["id"]), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # header text
        cv2.rectangle(frame, (10,10), (300,60), (0,0,0), -1)
        cv2.putText(frame, f"Occupied: {occupied}/{len(slots)}", (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        writer.write(frame)
        if SHOW:
            cv2.imshow("Parking YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # csv row
        row = {"frame": frame_idx, "time_s": round(frame_idx/FPS,3),
               "occupied": occupied, "total_slots": len(slots)}
        for i, s in enumerate(slots):
            row[f"slot_{s['id']}"] = int(slot_taken[i])
        rows.append(row)

    cap.release(); writer.release(); cv2.destroyAllWindows()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"[DONE] {frame_idx} frames in {time.time()-t0:.1f}s")
    print(f"- Video : {OUT_VIDEO}")
    print(f"- CSV   : {OUT_CSV}")

if __name__ == "__main__":
    main()