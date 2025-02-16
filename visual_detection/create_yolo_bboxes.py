import torch
import glob
import os
import cv2
import json
from ultralytics import YOLO

def process_images(model_path="./visual_detection/saved_models/yolov8n_custom_50epochs.pt",
                   input_folder="/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/",
                   output_folder="/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/",
                   blip_data_file="./vqa_generator/output_jsons/yolo_vqa_gt_data.json",
                   conf_threshold=0.5,
                   dig_threshold=30, 
                   hight_threshold=30):
    """
    Runs YOLOv8 inference, generates actions, and creates VQA pairs for BLIP training.

    Args:
        model_path (str): Path to the fine-tuned YOLOv8 model.
        input_folder (str): Directory containing images.
        output_folder (str): Directory to save processed images.
        blip_data_file (str): Path to save BLIP training data (Q&A pairs).
        conf_threshold (float): Confidence threshold for detections.
        dig_threshold (int): Distance threshold for digging decision.

    Returns:
        None
    """
    # Ensure MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load the fine-tuned YOLOv8 model
    model = YOLO(model_path).to(device)

    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

    # Define camera reference point (bottom center of the image)
    blip_data = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        vehicle_x = width // 2  # vehicle is assumed at the center bottom
        vehicle_y = height * 3 // 4

        # Run inference on a single image
        result = model(img, device=device)[0]  # Run inference and get first result

        # Initialize bbox count and action list
        n_bbox = 0

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                n_bbox += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Find closest bbox point to the vehicle point
                if abs(x1 - vehicle_x) < abs(x2 - vehicle_x):
                    closest_x = x1
                    farthest_x = x2
                else:
                    closest_x = x2
                    farthest_x = x1
                # closest_x = x1 if abs(x1 - vehicle_x) < abs(x2 - vehicle_x) else x2
                # farthest_x = x1 if abs(x1 - vehicle_x) > abs(x2 - vehicle_x) else x2
                closest_y = y1 if abs(y1 - vehicle_y) < abs(y2 - vehicle_y) else y2
                # direction action
                if closest_x < vehicle_x and closest_y < vehicle_y+hight_threshold:
                    direction = "left" 
                elif closest_x < vehicle_x and closest_y > vehicle_y+hight_threshold:
                    direction = "up-left"
                elif closest_x > vehicle_x and closest_y > vehicle_y+hight_threshold:
                    direction = "up-right"
                else:
                    direction = "right"

                # Determine digging action 
                if closest_y > vehicle_y:
                    if (closest_x > vehicle_x - dig_threshold and farthest_x > vehicle_x - dig_threshold) or (closest_x < vehicle_x - dig_threshold and farthest_x < vehicle_x - dig_threshold):
                        dig_action = "no dig"
                        dig__possible = "no, the pile is too far away"
                    else:
                        dig_action = "dig"
                        dig__possible = "yes"
                else:
                    dig_action = "no dig"
                    dig__possible = "no, the pile is too far away"
                bbox = {
                    "id": n_bbox,
                    "direction": direction,
                    "dig_action": dig_action, 
                    "dig_possible": dig__possible
                }

                # Draw bounding box and label
                label = f"{n_bbox})"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save processed image
                output_path = os.path.join(output_folder, os.path.basename(img_path))
                cv2.imwrite(output_path, img)

                # Generate VQA pairs for BLIP training
                questions = [
                    f"What direction should the vehicle go to reach pile {bbox['id']}?",
                    f"Should the excavator dig at bbox {bbox['id']}?",
                    f"Is digging possible for bbox {bbox['id']}?",
                ]
                answers = [f"{bbox['direction']}", 
                           f"{bbox['dig_action']}", 
                           f"{bbox['dig_possible']}"]

                for q, a in zip(questions, answers):
                    blip_data.append({
                        "image": os.path.basename(img_path),
                        "question": q,
                        "answer": a
                    })
        # for q, a in zip(questions, answers):
        blip_data.append({
                "image": os.path.basename(img_path),
                "question": [f"How many piles are there in the image?"],
                "answer": [f"{n_bbox}"]
            })

        # Print progress
        print(f"[{idx}/{len(image_paths)}] Processed: {os.path.basename(img_path)} → Saved to {output_folder}")

    # Save BLIP training data
    with open(blip_data_file, "w") as f:
        json.dump(blip_data, f, indent=4)

    print(f"\n✅ Processed {len(image_paths)} images. Results saved in {output_folder}")
    print(f"✅ BLIP training data saved in {blip_data_file}")

# Main function
if __name__ == "__main__":
    process_images()
