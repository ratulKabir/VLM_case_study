import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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
os.makedirs(vo_root, exist_ok=True)

# === Video folders ===
video_folders = ["video_2"]

for video in video_folders:
    print(f"\n=== Processing {video} ===")

    frames_path = os.path.join(frames_root, video)
    depth_path = os.path.join(depth_root, video)
    vo_output_path = os.path.join(vo_root, video)
    os.makedirs(vo_output_path, exist_ok=True)

    frames = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])

    # ORB Feature Detector
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    poses = [np.eye(4)]  # Initial pose

    for i in range(len(frames)-1):
        print(f"Processing frame {i+1}/{len(frames)-1}")

        img1 = cv2.imread(os.path.join(frames_path, frames[i]))
        img2 = cv2.imread(os.path.join(frames_path, frames[i+1]))

        depth1 = cv2.imread(os.path.join(depth_path, frames[i]), cv2.IMREAD_GRAYSCALE) / 255.0
        depth2 = cv2.imread(os.path.join(depth_path, frames[i+1]), cv2.IMREAD_GRAYSCALE) / 255.0

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Check if features were found
        if des1 is None or des2 is None:
            print(f"Skipping frame {i+1} — no features found.")
            poses.append(poses[-1])
            continue

        # Feature matching
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            print("Not enough good matches.")
            poses.append(poses[-1])
            continue

        pts3D = []
        pts2D = []

        for m in good_matches:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt

            Z = depth1[int(v1), int(u1)]

            if Z < 1e-3:
                continue

            X = (u1 - cx) * Z / fx
            Y = (v1 - cy) * Z / fy

            pts3D.append([X, Y, Z])
            pts2D.append([u2, v2])

        pts3D = np.array(pts3D, dtype=np.float32)
        pts2D = np.array(pts2D, dtype=np.float32)

        if len(pts3D) < 6:
            print("Not enough valid 3D points.")
            poses.append(poses[-1])
            continue

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3D, pts2D, K, distCoeffs=None, reprojectionError=3.0, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("PnP failed.")
            poses.append(poses[-1])
            continue

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.squeeze()

        last_pose = poses[-1]
        new_pose = last_pose @ np.linalg.inv(T)
        poses.append(new_pose)

    # === Done with video ===

    poses_np = np.stack(poses)  # (N, 4, 4)
    trajectory = poses_np[:, :3, 3]  # (N, 3)

    # Save outputs
    np.save(os.path.join(vo_output_path, "poses.npy"), poses_np)
    np.save(os.path.join(vo_output_path, "trajectory.npy"), trajectory)

    # Optional plot per video
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"Estimated Trajectory for {video}")
    plt.axis('equal')
    plt.grid()
    plt.show()

print("\n✅ Visual odometry results saved successfully!")
