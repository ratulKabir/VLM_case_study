import torch
import glob
import os
import cv2
from ultralytics import YOLO

def process_images(model_path = "./visual_detection/saved_models/yolov8n_custom_50epochs.pt",
                input_folder = "/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/",
                output_folder = "/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/", 
                conf_threshold=0.5):
    """
    Runs YOLOv8 inference on images in a folder and saves results with bounding boxes.

    Args:
        model_path (str): Path to the fine-tuned YOLOv8 model.
        input_folder (str): Directory containing images.
        output_folder (str): Directory to save processed images.
        conf_threshold (float): Confidence threshold for detections.

    Returns:
        None
    """
    # Ensure MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load the fine-tuned YOLOv8 model
    model = YOLO(model_path).to(device)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

    # Process images one by one to save memory
    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(img_path)

        # Run inference on a single image
        result = model(img, device=device)[0]  # Run inference and get first result

        # Draw bounding boxes for detections
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the image with detections
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, img)

        # Print progress message
        print(f"[{idx}/{len(image_paths)}] Processed: {os.path.basename(img_path)} → Saved to {output_folder}")

        # Clear memory (MPS doesn't use CUDA, but still good practice)
        torch.cuda.empty_cache()

    print(f"\n✅ Processed {len(image_paths)} images. Results saved in {output_folder}")

# Main function
if __name__ == "__main__":
    process_images()
