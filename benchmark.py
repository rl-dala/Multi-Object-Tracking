import time
import csv
from pathlib import Path

import torch
import cv2
from numpy import percentile, min, max
from ultralytics import YOLO

warmup_frames = 10
max_frames = 100
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt") # A pretrained YOLO11 model by Ultralytics
IMG_PATH = Path(r"./MOT16/train/MOT16-02/img1")
image_paths = sorted([i for i in IMG_PATH.iterdir() if i.is_file() and i.suffix == ".jpg"][:max_frames+warmup_frames])

if not image_paths:
    raise ValueError(f"No images found in {IMG_PATH}")
elif len(image_paths) <= warmup_frames:
    raise ValueError(f"Not enough images found in {IMG_PATH} for warmup. Found {len(image_paths)} frames, but need at least {warmup_frames+1}.")

print(f"Total amount of images: {len(image_paths)}, warmup frames: {warmup_frames}")

# Warmup 10 frames to avoid cold start latency
for i in image_paths[:warmup_frames]:
    image = cv2.imread(str(i))
    if image is None:
        continue
    results = model.predict(source=image,
                            device=device,
                            verbose=False)

frame_names = [] # List to store the names of the frames processed
time_per_inference = [] # List to store the inference time for each frame
time_per_prediction = [] # List to store the prediction time for each frame
frame_boxes_count = [] # List to store the number of bounding boxes for each frame

# Benchmarking the model on the remaining frames
for i in image_paths[warmup_frames:]:
    image = cv2.imread(str(i))
    if image is None:
        continue
    frame_names.append(i.name)
    if device != "cpu":
        torch.cuda.synchronize() # Synchronize CUDA to ensure accurate timing
        
    start = time.perf_counter()
    results = model.predict(source=image,
                            device=device,
                            verbose=False)
    
    if device != "cpu":
        torch.cuda.synchronize() # Synchronize CUDA
        
    time_per_prediction.append((time.perf_counter() - start) * 1000) # Full end-to-end prediction time in milliseconds
    time_per_inference.append(results[0].speed['inference']) # Inference time in milliseconds
    frame_boxes_count.append(len(results[0].boxes))

if len(time_per_prediction) == 0:
    raise ValueError("No frames were processed. Please check the input images and model.")

print(f"""
{len(time_per_prediction)} frames took {sum(time_per_prediction):.2f} ms
Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

Prediction latency results:
Prediction mean latency: {sum(time_per_prediction) / len(time_per_prediction):.2f} ms
Prediction median latency: {percentile(time_per_prediction, 50):.2f} ms
Prediction 95 percentile latency: {percentile(time_per_prediction, 95):.2f} ms
Prediction Min latency: {min(time_per_prediction):.2f} ms
Prediction Max latency: {max(time_per_prediction):.2f} ms

Inference latency results:
Inference mean latency: {sum(time_per_inference) / len(time_per_prediction):.2f} ms
Inference median latency: {percentile(time_per_inference, 50):.2f} ms
Inference 95 percentile latency: {percentile(time_per_inference, 95):.2f} ms
Inference Min latency: {min(time_per_inference):.2f} ms
Inference Max latency: {max(time_per_inference):.2f} ms

Average FPS: {1000 * len(time_per_prediction) / sum(time_per_prediction):.2f} fps

Note: The inference latency is the time taken by the model to process the input and produce the output, while the prediction latency includes additional overhead such as data loading and preprocessing.
""")

with open("benchmark_results.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "end_to_end_latency_ms", "inference_latency_ms", "num_boxes"])
    for f, e2e, inf, boxes in zip(frame_names, time_per_prediction, time_per_inference, frame_boxes_count):
        writer.writerow([f, e2e, inf, boxes])
print("Benchmark results saved to benchmark_results.csv")