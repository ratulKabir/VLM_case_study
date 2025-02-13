from data_generator import download_all_videos, process_videos

GEN_DATA = True

if __name__ == "__main__":
    # 1. Dataset generation
    if GEN_DATA:
        download_all_videos()
        process_videos()
        print("✅ All videos processed successfully!")

# 2. VQA pair generation
# 3. Object detection and position extraction
# 4. Integration and demonstration