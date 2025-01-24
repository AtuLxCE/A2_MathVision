# main.py

from utils.helpers import login_to_huggingface
from models.model_loader import load_model_and_tokenizer, configure_lora
from data.dataset import load_image_dataset, convert_to_conversation, split_dataset
from models.trainer import setup_trainer, run_inference
from config.settings import INSTRUCTION

def main():
    # Log in to Hugging Face
    login_to_huggingface()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Configure LoRA
    model = configure_lora(model)

    # Load dataset
    image_dataset = load_image_dataset()

    # Convert dataset to conversation format
    converted_image_dataset = [
        convert_to_conversation(sample, INSTRUCTION) for sample in image_dataset
    ]

    # Split dataset into train and validation sets
    train_data, val_data = split_dataset(converted_image_dataset, test_size=0.2, random_state=42)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Set up trainer with enhanced features
    trainer = setup_trainer(model, tokenizer, train_data, val_data, resume_from_checkpoint=None)

    # Train the model
    trainer.train()

    # Run inference
    # image = image_dataset[0]["dominated_image"]
    # run_inference(model, tokenizer, image, INSTRUCTION)

if __name__ == "__main__":
    main()