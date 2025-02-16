import torch
from ultralytics import YOLO

# Ensure MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the pre-trained YOLOv8 model (Nano version for speed)
model = YOLO("./visual_detection/saved_models/yolov8n.pt").to(device)  # Move model to MPS

# Train the model on your labeled data (few-shot learning)
model.train(data="./visual_detection/dataset.yaml", epochs=1, batch=4, device=device)

# Save the fine-tuned model
model.save("./visual_detection/saved_models/yolov8n_custom_1epochs.pt")

# Test the model on a new image
image_path = "/Users/ratul/Workstation/datasets/vlm/youtube/frames/dataset/images/test/frame_1059.jpg"
results = model(image_path)

# Show the detection results
results[0].show()
