from data_generator import download_all_videos, process_videos
from vqa_generator.vqa import generate_vqa
from visual_detection.create_yolo_bboxes import process_images

GEN_DATA = False
GEN_VQA_PAIRS = True
GEN_YOLO_BBOXES = False

if __name__ == "__main__":
    # 1. Dataset generation
    if GEN_DATA:
        download_all_videos()
        process_videos()
        print("✅ All videos processed successfully!")

    # 2. Generate bounding boxes for all images
    if GEN_YOLO_BBOXES:
        print("Generating YOLO bounding boxes...")
        process_images()
        print("YOLO bounding boxes generation complete!")

    # 3. VQA pair generation
    if GEN_VQA_PAIRS:
        print("Generating VQA pairs...")
        generate_vqa()
        print("VQA generation complete!")
    
    # 3. Object detection and position extraction
    
# 4. Integration and demonstration

    
