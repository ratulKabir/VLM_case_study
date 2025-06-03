import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# === Camera intrinsics ===
fx = fy = 600
cx = cy = 168
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# === Paths ===
base_path = "/home/ratul/Workstation/ratul/VLM_case_study/database"
frames_root = os.path.join(base_path, "frames")
depth_root = os.path.join(base_path, "depth")
vo_root = os.path.join(base_path, "vo")
bev_root = os.path.join(base_path, "bev")
os.makedirs(bev_root, exist_ok=True)

# === Video folders ===
video_folders = ["video_1", "video_2"]

# === BEV parameters ===
bev_resolution = 0.1  # meters per pixel (10cm grid)
bev_size = 100  # map size (100m x 100m)
bev_half_size = bev_size // 2

for video in video_folders:
    print(f"\n=== Processing BEV for {video} ===")

    frames_path = os.path.join(frames_root, video)
    depth_path = os.path.join(depth_root, video)
    vo_path = os.path.join(vo_root, video)
    bev_output_path = os.path.join(bev_root, video)
    os.makedirs(bev_output_path, exist_ok=True)

    frames = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
    poses = np.load(os.path.join(vo_path, "poses.npy"))  # (N, 4, 4)

    # Initialize BEV map as empty
    bev_map = np.zeros((bev_size, bev_size), dtype=np.uint8)

    for i, frame_file in enumerate(frames):
        if i % 50 == 0:
            print(f"Processing frame {i+1}/{len(frames)}")

        img = cv2.imread(os.path.join(frames_path, frame_file))
        depth = cv2.imread(os.path.join(depth_path, frame_file), cv2.IMREAD_GRAYSCALE) / 255.0

        H, W = depth.shape

        # Generate pixel grid
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        depth_flat = depth.flatten()

        # Backproject to camera 3D frame
        X = (u_flat - cx) * depth_flat / fx
        Y = (v_flat - cy) * depth_flat / fy
        Z = depth_flat

        cam_points = np.vstack((X, Y, Z)).T

        valid_mask = depth_flat > 1e-3
        cam_points = cam_points[valid_mask]

        cam_points_hom = np.hstack((cam_points, np.ones((cam_points.shape[0], 1))))  # (N, 4)

        world_points = (poses[i] @ cam_points_hom.T).T[:, :3]

        # Project onto BEV plane (XZ)
        X_world = world_points[:, 0]
        Z_world = world_points[:, 2]

        bev_x = np.round(X_world / bev_resolution).astype(int) + bev_half_size
        bev_z = np.round(Z_world / bev_resolution).astype(int) + bev_half_size

        valid_bev_mask = (bev_x >= 0) & (bev_x < bev_size) & (bev_z >= 0) & (bev_z < bev_size)
        bev_x = bev_x[valid_bev_mask]
        bev_z = bev_z[valid_bev_mask]

        bev_map[bev_z, bev_x] = 255  # directly update BEV map

    # === Save outputs ===
    np.save(os.path.join(bev_output_path, "bev_map.npy"), bev_map)

    bev_params = {
        "resolution": bev_resolution,
        "size": bev_size
    }
    with open(os.path.join(bev_output_path, "parameters.json"), "w") as f:
        json.dump(bev_params, f)

    # === Plot BEV for visual verification ===
    plt.figure(figsize=(6, 6))
    plt.imshow(bev_map, cmap='gray', origin='lower')
    plt.title(f"BEV Map for {video}")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid()
    plt.savefig(os.path.join(bev_output_path, "bev_map.png"))
    plt.show()

print("\n✅ BEV generation completed and saved successfully (RAM friendly).")
