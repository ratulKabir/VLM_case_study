from data_generator import download_all_videos, process_videos
from vqa_generator.vqa import generate_vqa
from vqa_generator import train_vlm, test_vlm
from visual_detection.create_yolo_bboxes import process_images
from utils.utils import save_image_qa_plot

GEN_DATA = False
GEN_VQA_PAIRS_USING_VLM = False
GEN_YOLO_BBOXES_AND_ACTION = False
TRAIN_VLM = False
TEST_VLM = True

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
    
    # 4. Train the VLM
    if TRAIN_VLM:
        print("Training the VLM...")
        train_vlm.train_model()
        print("VLM training complete!")    
    
    # 5. Demonstration
    if TEST_VLM:
        print("Testing the VLM...")
        # Dummy examples
        test_images = [
            ("/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/frame_1706.jpg", "Should the excavator dig at bbox 2?"),
            # ("/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/frame_0020.jpg", "How many piles?"),
            ("/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/frame_0123.jpg", "Should it dig for pile 1?"),
        ]

        for img_path, question in test_images:
            answer = test_vlm.test_model(img_path, question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
            save_image_qa_plot(img_path, question, answer, f"./results/{img_path.split('/')[-1].split('.')[0]}_output_plot.jpg")
        print("VLM testing complete!")

    
