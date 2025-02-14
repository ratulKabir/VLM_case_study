from data_generator import download_all_videos, process_videos
from vqa_generator.vqa import generate_vqa

GEN_DATA = False
GEN_VQA_PAIRS = True

if __name__ == "__main__":
    # 1. Dataset generation
    if GEN_DATA:
        download_all_videos()
        process_videos()
        print("✅ All videos processed successfully!")

    # 2. VQA pair generation
    if GEN_VQA_PAIRS:
        print("Generating VQA pairs...")
        generate_vqa()
        print("VQA generation complete!")
# 3. Object detection and position extraction
# 4. Integration and demonstration

    
