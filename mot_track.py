from collections import defaultdict
from pathlib import Path
import csv
import cv2

import torch
from ultralytics import YOLO

data_dir = Path("./MOT16/train")
output_dir = Path("./runs/mot_tracking")

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolo11n.pt")

sequences = sorted(path for path in data_dir.glob("*MOT16-*") if (path / "img1").is_dir())

if not sequences:
    raise FileNotFoundError(f"No MOT16 dataset found in {data_dir}")

for sequence in sequences:
    track_history = defaultdict(list)  # Dictionary to store track history for each frame, track ID and positions
    image_dir = sequence / "img1"
    image_paths = sorted(image_dir.glob("*.jpg"))
    
    if not image_paths:
        print(f"Skipping {sequence.name}: no images")
        continue
    
    print(f"Processing {sequence.name} with {len(image_paths)} images")
    
    if model.predictor is not None:
        for tracker in getattr(model.predictor, "trackers", []):
            tracker.reset()
    
    sequence_output_dir = output_dir / sequence.name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in image_paths:
        result = model.track(source=str(image_path),
                             device=device,
                             verbose=False,
                             persist=True,
                             tracker="bytetrack.yaml",
                             )[0]
        
        # Visualize the result on the frame
        image = result.plot(conf=False)
        output_file = sequence_output_dir / image_path.name
        
        if result.boxes and result.boxes.is_track:
            boxes_center = result.boxes.xywh.cpu()
            boxes = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            frame_number = int(image_path.stem)
            
            for box_corners, box_center, track_id in zip(boxes, boxes_center, track_ids):
                x1, y1, x2, y2 = box_corners
                x_center, y_center, w, h = box_center
                track = track_history[track_id]
                track.append((frame_number,
                              float(x1), float(y1),
                              float(x2), float(y2),
                              float(x_center), float(y_center)))  # x, y center point

        if not cv2.imwrite(str(output_file), image):
            raise RuntimeError(f"Failed to save image {output_file}")
        
    print(f"Saved image to {sequence_output_dir}")
    
    with open(sequence_output_dir / f"{sequence.name}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "track_id", 
                         "x1", "y1", "x2", "y2",
                         "center_x", "center_y"])
        for track_id, positions in track_history.items():
            for frame_number,x1, y1, x2, y2, x_center, y_center in positions:
                writer.writerow([frame_number, track_id,
                                 x1, y1, x2, y2,
                                 x_center, y_center])
    print(f"CSV results saved to {sequence_output_dir / sequence.name}.csv")