import numpy as np
import os
import json

# === Paths ===
base_path = "/home/ratul/Workstation/ratul/VLM_case_study/database"
vo_root = os.path.join(base_path, "vo")
measurement_root = os.path.join(base_path, "measurements")
perframe_root = os.path.join(base_path, "perframe_measurements")
os.makedirs(perframe_root, exist_ok=True)

# === Video folders ===
video_folders = ["video_1", "video_2"]

for video in video_folders:
    print(f"\n=== Processing per-frame measurements for {video} ===")

    vo_path = os.path.join(vo_root, video)
    measurement_path = os.path.join(measurement_root, video)
    perframe_output_path = os.path.join(perframe_root, video)
    os.makedirs(perframe_output_path, exist_ok=True)

    # Load ego trajectory
    poses = np.load(os.path.join(vo_path, "poses.npy"))
    num_frames = poses.shape[0]

    # Load global measurements
    with open(os.path.join(measurement_path, "measurements.json"), "r") as f:
        global_measurements = json.load(f)

    # Extract global object positions
    objects = [
        {
            "object_id": m["object_id"],
            "world_x": m["world_x"],
            "world_z": m["world_z"]
        }
        for m in global_measurements
    ]

    per_frame_data = {}

    for frame_idx in range(num_frames):
        ego_pose = poses[frame_idx]
        ego_x = ego_pose[0, 3]
        ego_z = ego_pose[2, 3]

        frame_measurements = []

        for obj in objects:
            rel_x = obj["world_x"] - ego_x
            rel_z = obj["world_z"] - ego_z
            distance = np.sqrt(rel_x**2 + rel_z**2)

            frame_measurements.append({
                "object_id": obj["object_id"],
                "relative_x": round(float(rel_x), 3),
                "relative_z": round(float(rel_z), 3),
                "distance": round(float(distance), 3)
            })

        per_frame_data[f"frame_{frame_idx:04d}"] = frame_measurements

    # Save per-frame measurements
    with open(os.path.join(perframe_output_path, "per_frame_measurements.json"), "w") as f:
        json.dump(per_frame_data, f, indent=4)

    print(f"Processed {num_frames} frames for {video}.")

print("\n✅ Per-frame measurement extraction completed and saved.")
