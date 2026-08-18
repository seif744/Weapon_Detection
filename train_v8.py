from ultralytics import YOLO

model = YOLO('yolov8m')

results = model.train(
    data='datasets/dataset.yaml',
    epochs=120,
    imgsz=640,
    name='yolo_weapons',
    workers=4,
    patience=25    
)
