import sys
from sahi import AutoDetectionModel
from sahi.predict import predict

model_path = sys.argv[1]
source = sys.argv[2]

det = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.25,
    device="cuda:0",
)

predict(
    detection_model=det,
    source=source,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    export_visual=True,
    project="runs/sahi",
    name="sahi_test",
)
