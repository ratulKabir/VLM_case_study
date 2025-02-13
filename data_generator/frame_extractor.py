import os
import subprocess
from .config_loader import config  # Import config

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    frame_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    command = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={config['fps']},scale={config['resolution']}",
        frame_pattern
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_videos():
    os.makedirs(config["frame_dir"], exist_ok=True)

    for i in range(len(config["video_links"])):
        video_filename = os.path.join(config["video_dir"], f"video_{i+1}.mp4")
        frame_output_folder = os.path.join(config["frame_dir"], f"video_{i+1}")

        print(f"Extracting and resizing frames from video {i+1}...")
        extract_frames(video_filename, frame_output_folder)

        print(f"Deleting video {i+1} to save space...")
        os.remove(video_filename)
