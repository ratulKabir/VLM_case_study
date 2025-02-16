from data_generator import download_all_videos, process_videos
from vqa_generator.vqa import generate_vqa
from visual_detection.create_yolo_bboxes import process_images

GEN_DATA = False
GEN_VQA_PAIRS_USING_VLM = False
GEN_YOLO_BBOXES_AND_ACTION = False

if __name__ == "__main__":
    # 1. Dataset generation
    if GEN_DATA:
        download_all_videos()
        process_videos()
        print("✅ All videos processed successfully!")

    # 2. VQA pair generation
    if GEN_VQA_PAIRS_USING_VLM:
        print("Generating VQA pairs...")
        generate_vqa()
        print("VQA generation complete!")

    # 3. Generate bounding boxes for all images
    if GEN_YOLO_BBOXES_AND_ACTION:
        print("Generating YOLO bounding boxes and actions to train the VLM...")
        process_images()
        print("YOLO bounding boxes generation complete!")
    
    # 3. Object detection and position extraction
    
# 4. Integration and demonstration

    
