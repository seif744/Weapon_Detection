import sys
from collections import defaultdict
from ultralytics import YOLO

model_path = sys.argv[1]
video      = sys.argv[2]
min_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 8
conf       = float(sys.argv[4]) if len(sys.argv) > 4 else 0.18

model = YOLO(model_path)
seen = defaultdict(int)
alerted = set()
frame = 0

for r in model.track(source=video, conf=conf, imgsz=1280, tracker="bytetrack.yaml",
                     persist=True, stream=True, device=0, verbose=False):
    frame += 1
    if r.boxes is None or r.boxes.id is None:
        continue
    for tid, cls in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
        seen[tid] += 1
        if seen[tid] >= min_frames and tid not in alerted:
            alerted.add(tid)
            print(f"[frame {frame}] ALERT: {model.names[cls]} (track {tid})")

print(f"\n--- {video}")
print(f"frames: {frame} | tracks seen: {len(seen)} | alerts fired: {len(alerted)}")
