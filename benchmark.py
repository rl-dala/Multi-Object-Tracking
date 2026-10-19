import time
import csv
from pathlib import Path

import torch
import cv2
from numpy import ndarray, percentile, min, max
from ultralytics import YOLO

print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

IMG_PATH = Path(r"./MOT16/train/MOT16-02/img1")
warmup_frames = 10
max_frames = 100
image_paths = sorted([i for i in IMG_PATH.iterdir() if i.is_file() and i.suffix == ".jpg"][:max_frames+warmup_frames])
num_of_frames = len(image_paths) - warmup_frames
print(f"Total amount of images: {num_of_frames + warmup_frames}")

model = YOLO("yolo11n.pt")
for i in image_paths[:warmup_frames]:
    results = model.predict(source=i, device=device, verbose=False)
frame_names = [i.name for i in image_paths[warmup_frames:]]
time_per_inference = []
time_per_frame = []
frame_boxes = []

for i in image_paths[warmup_frames:]:
    image = cv2.imread(str(i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    results = model.predict(source=image, device=device, verbose=False)
    time_per_frame.append((time.perf_counter() - start) * 1000)
    time_per_inference.append(results[0].speed['inference'])
    frame_boxes.append(results[0].boxes)

print("""
      """)
print(f"Time take for {num_of_frames} frames: {sum(time_per_frame):.2f} ms")

print(f"Mean latency: mean_latency_ms={sum(time_per_frame) / num_of_frames:.2f} ms")
print(f"Median latency: p50_latency_ms={percentile(time_per_inference, 50):.2f} ms")
print(f"95 percentile latency: p95_latency_ms={percentile(time_per_inference, 95):.2f} ms")
print(f"Min latency: min_latency_ms={min(time_per_inference):.2f} ms")
print(f"Max latency: max_latency_ms={max(time_per_inference):.2f} ms")
print(f"FPS: {1000 * num_of_frames / sum(time_per_frame):.2f} fps")

with open("benchmark_results.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "end_to_end_latency_ms", "inference_latency_ms"])
    for f, e2e, inf in zip(frame_names, time_per_frame, time_per_inference):
        writer.writerow([f, e2e, inf])