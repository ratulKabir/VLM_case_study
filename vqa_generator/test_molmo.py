from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import torch
from PIL import Image

# Load model and processor
model_name = "allenai/Molmo-7B-D-0924"
# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto', 
)

# Load an image
image = Image.open("/Users/ratul/Workstation/datasets/vlm/youtube/frames/video_1/frame_0101.jpg")

# Define the question
question = "Where is the pile, left right, ahead or no-pile?"

# Prepare inputs
inputs = processor(images=image, text=question, return_tensors="pt")

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("Answer:", generated_text)
