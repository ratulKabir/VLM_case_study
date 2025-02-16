import os
import json
import torch.backends
import yaml
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, Blip2Processor, Blip2ForConditionalGeneration

# Load config from YAML file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_config():
    """Load configuration from config.yaml"""
    with open(CONFIG_FILE, "r") as file:
        return yaml.safe_load(file)

def load_model(model_type):
    """Load the selected model dynamically based on the config"""
    if model_type == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE, dtype=torch.float16)
    elif model_type == "blip2":
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # load the model weights in float16 instead of float32
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return processor, model

def generate_answer(image_path, question, processor, model, model_type):
    """Generate answers using BLIP or CLIP based on the image and question"""
    image = Image.open(image_path).convert("RGB")

    if model_type == "blip" or model_type == "blip2":
        # text_prompt = f"Based on the image, {question}"  # More natural phrasing
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE, dtype=torch.float16)
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True).strip()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def create_vqa_pairs(image_folder, questions, processor, model, model_type):
    """Generate VQA pairs dynamically based on both the image and the question"""
    vqa_data = []
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files[:100]:  # Process a few images
        img_path = os.path.join(image_folder, img_file)

        vqa_pairs = [{"question": q, "answer": generate_answer(img_path, q, processor, model, model_type)} for q in questions]
        # for q in questions:
        #     # q = "Question: " + q + " Answer:"
        #     answer = generate_answer(img_path, q, processor, model, model_type)
        #     vqa_pairs = [{"question": q, "answer": answer}]

        vqa_data.append({"image": img_file, "vqa_pairs": vqa_pairs})

    return vqa_data

def generate_vqa():
    """Main function to generate and save VQA pairs"""
    config = load_config()

    # Extract global parameters
    IMAGE_FOLDER = config["global"]["image_folder"]
    MODEL_TYPE = config["global"]["model_type"]
    config["global"]["output_file"] = f"./vqa_generator/output_jsons/vqa_pairs_{MODEL_TYPE}.json"
    OUTPUT_FILE = config["global"]["output_file"]


    # Load questions from config
    questions_list = config["questions"]

    # Load selected model
    processor, model = load_model(MODEL_TYPE)

    # Generate and save VQA pairs
    vqa_results = create_vqa_pairs(IMAGE_FOLDER, questions_list, processor, model, MODEL_TYPE)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(vqa_results, f, indent=4)

    print(f"VQA pairs generated and saved to {OUTPUT_FILE}")
