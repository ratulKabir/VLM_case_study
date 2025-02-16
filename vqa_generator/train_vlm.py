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
    answer_encoding = processor.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True)
    
    inputs["labels"] = answer_encoding["input_ids"].squeeze(0)
    inputs["labels_attention_mask"] = answer_encoding["attention_mask"].squeeze(0)  # Ensure mask is included

    return {key: val.squeeze(0) for key, val in inputs.items()}


def train_model(
    data_path="/Users/ratul/Workstation/github_repos/VLM_case_study/vqa_generator/output_jsons/yolo_vqa_gt_data.json",
    output_dir="./vqa_generator/saved_model/blip_finetuned",
    num_epochs=50,
    learning_rate=5e-4,
    batch_size=4
):
    """Fine-tune BLIP model for Visual Question Answering"""
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load BLIP model and processor
    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

    # Load JSON file
    with open(data_path, "r") as f:
        data = json.load(f)

    # Convert into Hugging Face Dataset format
    dataset = Dataset.from_list(data).train_test_split(test_size=0.1)

    # Apply preprocessing while filtering out failed cases
    dataset = dataset.map(lambda example: preprocess_data(example, processor), remove_columns=["image", "question", "answer"]).filter(lambda x: x is not None)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        push_to_hub=False,
    )

    # Freeze all layers except the classifier head for faster training
    for name, param in model.named_parameters():
        if "text_decoder.cls.predictions" not in name:  # Keep only this part trainable
            param.requires_grad = False

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # Fine-tune the model
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train_model()
