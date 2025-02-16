from ultralytics import YOLO

# Load the fine-tuned YOLOv8 model
model = YOLO("./visual_detection/saved_models/yolov8n_custom.pt")  # Or "runs/train/exp/weights/best.pt"

img_paths = ['/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_2024.jpg',
             '/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_1059.jpg',
             '/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_1060.jpg',
             '/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_0055.jpg',
             '/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_0960.jpg',]

# Run inference on a test image
results = model(img_paths)

# Show detection results
results.show()