'''
Ref: https://huggingface.co/docs/transformers/tasks/mono_depth_estimation
'''

import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "Intel/zoedepth-nyu-kitti"
# checkpoint = "intel-isl/MiDaS"

device = DEVICE
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)


image = read_image("/home/ratul/Workstation/ratul/VLM_case_study/database/frames/video_1/frame_0822.jpg")
pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
    output = model(pixel_values)

# ZoeDepth dynamically pads the input image. Thus we pass the original image size as argument
# to `post_process_depth_estimation` to remove the padding and resize to original dimensions.
post_processed_output = image_processor.post_process_depth_estimation(
    output,
    source_sizes=[(image.shape[-2], image.shape[-1])],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
depth = depth.astype("uint8")

# Plot using matplotlib.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.detach().cpu().numpy().transpose(1, 2, 0).astype("uint8"))  # Convert to HWC format for plotting
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap='inferno')  # or 'plasma', 'magma'
plt.title("Estimated Depth")
plt.colorbar()

plt.savefig("./results/depth_est/depth_estimation_output_zoe_0.jpg")