from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = "model/yolo26n-seg.pt"  # Your model path
VIDEO_PATH = "data/video.mp4"  # Your video path
OUTPUT_DIR = "output"
SHOW_PREVIEW = True  # Set to True to see real-time preview

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load the YOLO segmentation model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Open video file
print(f"Processing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video from {VIDEO_PATH}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")
print(f"  Duration: {total_frames / fps:.2f} seconds")

# Prepare output video writer
output_path = f"{OUTPUT_DIR}/segmented_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
objects_detected_per_frame = []

print("\nProcessing video frames...")
print("Press 'q' to stop processing early (if preview is enabled)")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform segmentation on the frame
    results = model(frame, verbose=False)  # verbose=False to reduce console output

    # Get annotated frame with segmentation masks
    annotated_frame = results[0].plot()

    # Write to output video
    out.write(annotated_frame)

    # Count objects detected in this frame
    if results[0].masks is not None:
        num_objects = len(results[0].masks)
        objects_detected_per_frame.append(num_objects)
    else:
        objects_detected_per_frame.append(0)

    # Show preview if enabled
    if SHOW_PREVIEW:
        cv2.imshow('YOLO Video Segmentation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessing interrupted by user")
            break

    frame_count += 1

    # Progress update every 30 frames
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        avg_objects = np.mean(objects_detected_per_frame[-30:])
        print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) | Avg objects: {avg_objects:.1f}")

# Release resources
cap.release()
out.release()
if SHOW_PREVIEW:
    cv2.destroyAllWindows()

# Print summary statistics
print("\n" + "=" * 50)
print("VIDEO SEGMENTATION COMPLETE!")
print("=" * 50)
print(f"Output saved to: {output_path}")
print(f"Frames processed: {frame_count}/{total_frames}")
print(f"\nDetection Statistics:")
print(f"  Total objects detected: {sum(objects_detected_per_frame)}")
print(f"  Average objects per frame: {np.mean(objects_detected_per_frame):.2f}")
print(f"  Max objects in a frame: {max(objects_detected_per_frame) if objects_detected_per_frame else 0}")
print(f"  Min objects in a frame: {min(objects_detected_per_frame) if objects_detected_per_frame else 0}")
print(f"  Frames with detections: {sum(1 for x in objects_detected_per_frame if x > 0)}")
print("=" * 50)