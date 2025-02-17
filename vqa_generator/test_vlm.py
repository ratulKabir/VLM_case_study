import torch
from transformers import BlipForQuestionAnswering, BlipProcessor
from PIL import Image

# Load fine-tuned model and processor
checkpoint_path = "/Users/ratul/Workstation/github_repos/VLM_case_study/vqa_generator/saved_model/blip_finetuned/checkpoint-1645"
model = BlipForQuestionAnswering.from_pretrained(checkpoint_path, local_files_only=True)
checkpoint = "Salesforce/blip-vqa-base"  # Use the original BLIP model
processor = BlipProcessor.from_pretrained(checkpoint)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def test_model(image_path, question):
    """Pass an image and question to the fine-tuned model"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Preprocess the input
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    # Get model output
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode answer
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # Dummy examples
    test_images = [
        ("/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/frame_1105.jpg", "Should the excavator dig at bbox 1?"),
        ("/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1_yolo_bboxes/frame_1702.jpg", "How many piles are there in the image?"),
    ]

    for img_path, question in test_images:
        answer = test_model(img_path, question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
