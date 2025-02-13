import yt_dlp
import os
from .config_loader import config  # Import config

def download_video(url, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_all_videos():
    os.makedirs(config["video_dir"], exist_ok=True)

    for i, link in enumerate(config["video_links"]):
        video_filename = os.path.join(config["video_dir"], f"video_{i+1}.mp4")
        print(f"Downloading video {i+1} from {link}...")
        download_video(link, video_filename)
