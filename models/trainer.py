from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback
import os

class UploadToHuggingFaceCallback(TrainerCallback):
    def __init__(self, model, tokenizer, hf_repo_name):
        self.model = model
        self.tokenizer = tokenizer
        self.hf_repo_name = hf_repo_name

    def on_train_end(self, args, state, control, **kwargs):
        # Upload the best model to Hugging Face
        self.model.push_to_hub(self.hf_repo_name)
        self.tokenizer.push_to_hub(self.hf_repo_name)

def setup_trainer(image_model, image_tokenizer, train_dataset, val_dataset, hf_repo_name):
    # Define the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)

    # Define the callback for uploading the best model to Hugging Face
    upload_to_hf_callback = UploadToHuggingFaceCallback(image_model, image_tokenizer, hf_repo_name)

    # Define WandB settings
    wandb_config = {
        "project": "A2_ImageModel_200Epoch",  # WandB project name
        "name": "run-1",  # WandB run name
        "resume": "allow",  # Allow resuming runs
        "id": None,  # Set to a specific run ID to resume
    }

    trainer = SFTTrainer(
        model=image_model,
        tokenizer=image_tokenizer,
        data_collator=UnslothVisionDataCollator(image_model, image_tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            max_steps=200,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="output/",
            report_to="wandb",  # Report to WandB
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
            save_strategy="steps",
            save_steps=10,
            evaluation_strategy="steps",
            eval_steps=10,
            load_best_model_at_end=True,  # Load the best model at the end of training
            metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
            greater_is_better=False,  # Lower evaluation loss is better
        ),
        callbacks=[early_stopping_callback, upload_to_hf_callback],  # Add callbacks
    )
    return trainer