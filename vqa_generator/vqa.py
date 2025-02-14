import os
import json
import yaml
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load config from YAML file
CONFIG_FILE = "config.yaml"

def load_config():
    with open(os.path.join(os.path.dirname(__file__), CONFIG_FILE), "r") as file:
        return yaml.safe_load(file)

# Function to generate captions
def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to create VQA pairs using loaded questions
def create_vqa_pairs(image_folder, questions, processor, model):
    vqa_data = []
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files[:3]:  # Process only a few images
        img_path = os.path.join(image_folder, img_file)
        caption = generate_caption(img_path, processor, model)

        vqa_pairs = [{"question": q, "answer": caption} for q in questions]

        vqa_data.append({"image": img_file, "vqa_pairs": vqa_pairs})

    return vqa_data

# # Load the full config
# config = load_config()

# # Extract global parameters
# IMAGE_FOLDER = config["global"]["image_folder"]
# OUTPUT_FILE = config["global"]["output_file"]
# MODEL_NAME = config["global"]["model_name"]

# # Load questions from config
# questions_list = config["questions"]

# # Load BLIP model dynamically from config
# processor = BlipProcessor.from_pretrained(MODEL_NAME)
# model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

# # Generate and save VQA pairs
# vqa_results = create_vqa_pairs(IMAGE_FOLDER, questions_list, processor, model)
# with open(OUTPUT_FILE, "w") as f:
#     json.dump(vqa_results, f, indent=4)

# print(f"VQA pairs generated and saved to {OUTPUT_FILE}")


def generate_vqa():
    config = load_config()

    # Extract global parameters
    IMAGE_FOLDER = config["global"]["image_folder"]
    OUTPUT_FILE = config["global"]["output_file"]
    MODEL_NAME = config["global"]["model_name"]

    # Load questions from config
    questions_list = config["questions"]

    # Load BLIP model dynamically from config
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Generate and save VQA pairs
    vqa_results = create_vqa_pairs(IMAGE_FOLDER, questions_list, processor, model)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(vqa_results, f, indent=4)

    print(f"VQA pairs generated and saved to {OUTPUT_FILE}")