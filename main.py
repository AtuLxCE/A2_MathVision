import wandb
from utils.helpers import login_to_huggingface
from models.model_loader import load_model_and_tokenizer, configure_lora
from data.dataset import load_image_dataset, convert_to_conversation, split_dataset
from models.trainer import setup_trainer
from config.settings import INSTRUCTION, HUGGINGFACE_HUB_REPO_NAME

def main():
    # Log in to Hugging Face
    login_to_huggingface()

    # Initialize WandB
    wandb.init(
        project="A2_ImageModel_200Epoch",  # WandB project name
        name="run-1",  # WandB run name
        resume="allow",  # Allow resuming runs
        id=None,  # Set to a specific run ID to resume
    )

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

    # Set up trainer with enhanced features
    trainer = setup_trainer(
        model,
        tokenizer,
        train_data,
        val_data,
        hf_repo_name=HUGGINGFACE_HUB_REPO_NAME,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()