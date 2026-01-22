from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = "model/yolo26n-seg.pt"  # Your model path
IMAGE_PATH = "data/image.png"  # Your image path
OUTPUT_DIR = "output"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load the YOLO segmentation model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Read the image
print(f"Processing image: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: Could not read image from {IMAGE_PATH}")
else:
    # Perform segmentation
    results = model(image)

    # Process results
    for idx, result in enumerate(results):
        # Get the annotated image with segmentation masks
        annotated_img = result.plot()

        # Save the result
        output_path = f"{OUTPUT_DIR}/segmented_output_{idx}.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"Saved segmented image to: {output_path}")

        # Print detection information
        if result.masks is not None:
            print(f"\nDetected {len(result.masks)} objects:")
            for i, (box, mask, cls) in enumerate(zip(result.boxes, result.masks, result.boxes.cls)):
                class_name = model.names[int(cls)]
                confidence = box.conf[0]
                print(f"  {i + 1}. {class_name} (confidence: {confidence:.2f})")
        else:
            print("No objects detected with segmentation masks")

    print("\nSegmentation complete!")