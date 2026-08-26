from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data='datasets_merged/dataset.yaml',
    epochs=80,
    imgsz=960,
    batch=8,
    device=0,
    workers=8,
    degrees=10,
    perspective=0.0005,
    scale=0.6,
    translate=0.15,
    fliplr=0.5,
    hsv_v=0.5,
    erasing=0.4,
    mosaic=1.0,
    close_mosaic=15,
    patience=20,
    name='merged_v1',
    exist_ok=True,
)
