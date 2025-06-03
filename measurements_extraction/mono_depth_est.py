'''
Ref: https://huggingface.co/docs/transformers/tasks/mono_depth_estimation
'''

import os
import cv2
import torch
from torchvision.io import read_image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "Intel/zoedepth-nyu-kitti"

# Load model & processor
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(DEVICE)

# Root folder (adjust if needed)
root_folder = "/home/ratul/Workstation/ratul/VLM_case_study/database/frames"
output_root = "/home/ratul/Workstation/ratul/VLM_case_study/database/depth"

# Process both video folders
video_folders = ["video_1", "video_2"]

for video in video_folders:
    input_path = os.path.join(root_folder, video)
    output_path = os.path.join(output_root, video)
    os.makedirs(output_path, exist_ok=True)

    files = sorted(os.listdir(input_path))

    for idx, file in enumerate(files):
        if not file.endswith(('.jpg', '.png')):  # skip non-image files
            continue

        image_path = os.path.join(input_path, file)
        image = read_image(image_path)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(DEVICE)

        with torch.no_grad():
            output = model(pixel_values)

        post_processed_output = image_processor.post_process_depth_estimation(
            output,
            source_sizes=[(image.shape[-2], image.shape[-1])],
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth_min = predicted_depth.min()
        depth_max = predicted_depth.max()
        depth_range = depth_max - depth_min + 1e-8  # stability
        depth = (predicted_depth - depth_min) / depth_range
        depth = depth.detach().cpu().numpy() * 255
        depth = depth.astype("uint8")

        # Save depth image using OpenCV
        out_file = os.path.join(output_path, file)
        cv2.imwrite(out_file, depth)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(files)} frames for {video}")

print("✅ All depth maps generated successfully.")
