import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, AutoProcessor, AutoModelForVision2Seq, LlavaProcessor, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import open_clip
import requests
from io import BytesIO

def load_image(image_path):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

# BLIP Model
class BLIP:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    def query(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)

# BLIP-2 Model
class BLIP2:
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    def query(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)

# OpenCLIP Model (Text-Image Matching)
class OpenCLIP:
    def __init__(self):
        self.model, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    def query(self, image, question):
        image = self.preprocess(image).unsqueeze(0)
        text = self.tokenizer([question])
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            similarity = (image_features @ text_features.T).item()
        
        return f"Similarity Score: {similarity:.4f} (higher means better match)"

# Flamingo Model
class Flamingo:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/Flamingo")
        self.model = AutoModelForVision2Seq.from_pretrained("openai/Flamingo")
    
    def query(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)

# LLaVA Model
class Llava:
    def __init__(self):
        self.processor = LlavaProcessor.from_pretrained("liuhaotian/llava-7b")
        self.model = LlavaForConditionalGeneration.from_pretrained("liuhaotian/llava-7b")
    
    def query(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)

# Run comparison
if __name__ == "__main__":
    image_path = "/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_0101.jpg"  # Change this to your image path or URL
    question = "Is there a pile?"
    
    image = load_image(image_path)
    
    blip = BLIP()
    blip2 = BLIP2()
    clip = OpenCLIP()
    flamingo = Flamingo()
    llava = Llava()
    
    blip_answer = blip.query(image, question)
    blip2_answer = blip2.query(image, question)
    clip_answer = clip.query(image, question)
    flamingo_answer = flamingo.query(image, question)
    llava_answer = llava.query(image, question)
    
    print("--- Model Comparisons ---")
    print(f"BLIP Answer: {blip_answer}")
    print(f"BLIP-2 Answer: {blip2_answer}")
    print(f"OpenCLIP Score: {clip_answer}")
    print(f"Flamingo Answer: {flamingo_answer}")
    print(f"LLaVA Answer: {llava_answer}")
