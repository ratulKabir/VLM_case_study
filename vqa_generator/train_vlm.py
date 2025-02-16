import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
import json

def preprocess_data(example, processor):
    """Preprocess dataset examples"""
    try:
        image = Image.open(example["image"]).convert("RGB")  # Load image
    except Exception as e:
        print(f"Error loading image {example['image']}: {e}")
        return None  # Skip problematic images
    
    question = example["question"]
    answer = example["answer"]
    
    inputs = processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True)
    # inputs["labels"] = processor.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True)["input_ids"]
    answer_encoding = processor.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True)
    inputs["labels"] = answer_encoding["input_ids"].squeeze(0)
    inputs["labels_attention_mask"] = answer_encoding["attention_mask"].squeeze(0)  # Ensure mask is included

    
    return {key: val.squeeze(0) for key, val in inputs.items()}

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP model and processor
model_name = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

# Load JSON file
with open("/Users/ratul/Workstation/github_repos/VLM_case_study/vqa_generator/output_jsons/yolo_vqa_gt_data.json", "r") as f:
    data = json.load(f)

# Convert into Hugging Face Dataset format
dataset = Dataset.from_list(data).train_test_split(test_size=0.1)

# Apply preprocessing while filtering out failed cases
dataset = dataset.map(lambda example: preprocess_data(example, processor), remove_columns=["image", "question", "answer"]).filter(lambda x: x is not None)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vqa_generator/saved_model/blip_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=30,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./vqa_generator/saved_model/logs",
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./vqa_generator/saved_model")
