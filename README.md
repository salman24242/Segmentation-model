YOLO Image Segmentation Project:
- A simple Python project for performing instance segmentation on images using YOLO (You Only Look Once) segmentation models.

Overview:
- This project uses YOLOv26 segmentation models to detect, classify, and segment objects in images with pixel-level precision. The model provides bounding boxes, class labels, confidence scores, and segmentation masks for each detected object.


Features:
1) Object Detection: Identifies and locates objects with bounding boxes
2) Classification: Recognizes object classes (person, car, giraffe, etc.)
3) Instance Segmentation: Provides pixel-precise masks for each detected object
4) Confidence Scores: Shows detection confidence for each object
5) Easy to Use: Simple script that processes images with minimal configuration


Clone the repository

- git clone https://github.com/yourusername/yolo_project_segmentation.git
- cd yolo_project_segmentation

Create a virtual environment

- python -m venv venv

Activate the virtual environment

- venv\Scripts\activate

Install required packages

- pip install ultralytics opencv-python pillow numpy

1) Place your image in the data/ folder (default: image.png)
2) Place your YOLO model in the model/ folder (default: yolo26n-seg.pt)
3) Run the segmentation script

- python yolo_segment.py

Segmentation complete!

Technologies Used
1) Ultralytics YOLO: State-of-the-art object detection and segmentation
2) OpenCV: Image processing and manipulation
3) NumPy: Numerical computing
4) Python: Programming language

Model Information:
This project uses YOLO segmentation models that perform:

1) Object detection with bounding boxes
2) Multi-class classification
3) Instance segmentation with pixel-level masks
