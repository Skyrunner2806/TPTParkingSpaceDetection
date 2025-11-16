import cv2
import json
import numpy as np
from pathlib import Path


VIDEO_PATH = "/home/skyrunner/Documents/testing/video_new.mp4"
OUTPUT_JSON = "/home/skyrunner/Documents/testing/slots2.json"
FRAME_INDEX = 0 
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Gagal membuka video: {VIDEO_PATH}")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"Gagal membaca frame dari {VIDEO_PATH}")

H, W = frame.shape[:2]
slots = []
current_points = []
show_help = True

HELP_TEXT = """\
[Parking Slot Annotator]
Klik kiri  : tambah titik
ENTER / c  : tutup poligon
u          : undo titik terakhir
d          : hapus slot terakhir
s          : simpan ke slots.json
q / ESC    : keluar
"""

def draw_help(img):
    y0 = 110
    pad = 24
    lines = HELP_TEXT.strip().splitlines()
    cv2.rectangle(img, (10, y0-20), (450, y0+pad*len(lines)+10), (0,0,0), -1)
    for i, line in enumerate(lines):
        y = y0 + i*pad
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def draw_overlay(img, slots, current_points, show_help=True):
    disp = img.copy()
    for i, slot in enumerate(slots, start=1):
        pts = np.array(slot["points"], np.int32)
        cv2.polylines(disp, [pts], True, (0,255,0), 2)
        cx = int(np.mean(pts[:,0])); cy = int(np.mean(pts[:,1]))
        cv2.putText(disp, str(i), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if current_points:
        pts = np.array(current_points, np.int32)
        cv2.polylines(disp, [pts], False, (0,165,255), 2)
        for (x,y) in current_points:
            cv2.circle(disp, (x,y), 4, (0,165,255), -1)
    if show_help:
        draw_help(disp)
    return disp

def save_json():
    data = {
        "meta": {
            "video_path": VIDEO_PATH,
            "frame_index": FRAME_INDEX,
            "image_size": [H, W]
        },
        "slots": [{"id": i+1, "points": slot["points"]} for i, slot in enumerate(slots)]
    }
    json.dump(data, open(OUTPUT_JSON, "w"), indent=2)
    print(f"[✅] Disimpan ke: {OUTPUT_JSON}")

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([int(x), int(y)])

cv2.namedWindow("Annotate Parking Slots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotate Parking Slots", min(W,1280), min(H,720))
cv2.setMouseCallback("Annotate Parking Slots", mouse_event)

print("Instruksi:\n" + HELP_TEXT)
print(f"Frame dari: {VIDEO_PATH}")

while True:
    disp = draw_overlay(frame, slots, current_points, show_help)
    cv2.imshow("Annotate Parking Slots", disp)
    key = cv2.waitKey(30) & 0xFF

    if key in (13, ord('c')):  
        if len(current_points) >= 3:
            slots.append({"points": current_points.copy()})
            current_points.clear()
        else:
            print("Minimal 3 titik untuk 1 slot!")

    elif key == ord('u'): 
        if current_points:
            current_points.pop()
            print("Undo titik terakhir.")
        else:
            print("Tidak ada titik untuk dihapus.")

    elif key == ord('d'): 
        if slots:
            slots.pop()
            print("Slot terakhir dihapus.")
        else:
            print("Belum ada slot.")

    elif key == ord('s'):
        save_json()

    elif key in (ord('q'), 27):
        save_json()
        break

cv2.destroyAllWindows()