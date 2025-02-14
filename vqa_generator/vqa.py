import os
import json
import yaml
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip
from torchvision import transforms

# Load config from YAML file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    """Load configuration from config.yaml"""
    with open(CONFIG_FILE, "r") as file:
        return yaml.safe_load(file)

def load_model(model_type):
    """Load the selected model dynamically based on the config"""
    if model_type == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    elif model_type == "clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = clip.load("ViT-B/32", device=device)  # Load CLIP
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return processor, model

def generate_caption(image_path, processor, model, model_type):
    """Generate captions using BLIP or CLIP"""
    if model_type == "blip":
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)

    elif model_type == "clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)),
        ])
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            text_inputs = clip.tokenize(["A photo of a construction site", "A loader", "A pile of dirt"]).to(device)
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        best_match = ["Construction site", "Loader", "Pile of dirt"][similarity.argmax().item()]
        return f"The image likely contains: {best_match}"

def create_vqa_pairs(image_folder, questions, processor, model, model_type):
    """Generate VQA pairs dynamically"""
    vqa_data = []
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files[:3]:  # Process a few images
        img_path = os.path.join(image_folder, img_file)
        caption = generate_caption(img_path, processor, model, model_type)

        vqa_pairs = [{"question": q, "answer": caption} for q in questions]

        vqa_data.append({"image": img_file, "vqa_pairs": vqa_pairs})

    return vqa_data

def generate_vqa():
    """Main function to generate and save VQA pairs"""
    config = load_config()

    # Extract global parameters
    IMAGE_FOLDER = config["global"]["image_folder"]
    OUTPUT_FILE = config["global"]["output_file"]
    MODEL_TYPE = config["global"]["model_type"]

    # Load questions from config
    questions_list = config["questions"]

    # Load selected model
    processor, model = load_model(MODEL_TYPE)

    # Generate and save VQA pairs
    vqa_results = create_vqa_pairs(IMAGE_FOLDER, questions_list, processor, model, MODEL_TYPE)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(vqa_results, f, indent=4)

    print(f"VQA pairs generated and saved to {OUTPUT_FILE}")
