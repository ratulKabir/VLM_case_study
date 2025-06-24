import cv2
import numpy as np
import os
import json

# === Paths ===
base_path = "/home/ratul/Workstation/ratul/VLM_case_study/database"
bev_root = os.path.join(base_path, "bev")
measurement_root = os.path.join(base_path, "measurements")
os.makedirs(measurement_root, exist_ok=True)

# === Video folders ===
video_folders = ["video_1", "video_2"]

for video in video_folders:
    print(f"\n=== Processing measurements for {video} ===")

    bev_output_path = os.path.join(bev_root, video)
    measurement_output_path = os.path.join(measurement_root, video)
    os.makedirs(measurement_output_path, exist_ok=True)

    # Load BEV map
    bev_map = np.load(os.path.join(bev_output_path, "bev_map.npy"))

    # Load BEV parameters
    with open(os.path.join(bev_output_path, "parameters.json"), "r") as f:
        params = json.load(f)
    resolution = params["resolution"]
    size = params["size"]
    bev_half_size = size // 2

    # Convert BEV to binary mask
    _, binary_map = cv2.threshold(bev_map, 127, 255, cv2.THRESH_BINARY)

    # Connected components to find objects
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)

    measurements = []
    object_id = 0

    for i in range(1, num_labels):  # Skip label 0 (background)
        cx_bev, cz_bev = centroids[i]

        # Convert BEV pixels to world meters
        world_x = (cx_bev - bev_half_size) * resolution
        world_z = (cz_bev - bev_half_size) * resolution

        distance = np.sqrt(world_x**2 + world_z**2)

        measurements.append({
            "object_id": object_id,
            "world_x": round(float(world_x), 3),
            "world_z": round(float(world_z), 3),
            "distance": round(float(distance), 3),
            "bev_x": int(cx_bev),
            "bev_z": int(cz_bev)
        })

        object_id += 1

    # Save measurements to JSON
    with open(os.path.join(measurement_output_path, "measurements.json"), "w") as f:
        json.dump(measurements, f, indent=4)

    print(f"Found {len(measurements)} objects in {video}.")

print("\n✅ Measurement extraction completed and saved successfully.")
