# models/trainer.py

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from transformers import TextStreamer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
from transformers import TrainerState, TrainerControl
from transformers import TrainingArguments
from config.settings import (
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS,
    MAX_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_SCHEDULER_TYPE,
    SEED,
    OUTPUT_DIR,
    REPORT_TO,
    MAX_SEQ_LENGTH,
)

class CustomLoggingCallback(TrainerCallback):
    """Custom callback for logging training progress."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(f"Step {state.global_step}: {logs}")

def setup_trainer(model, tokenizer, train_dataset, eval_dataset=None, resume_from_checkpoint=None):
    """Set up the SFTTrainer with enhanced features."""
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,  # Log every 10 steps
        save_steps=10,  # Save a checkpoint every 10 steps
        save_total_limit=5,  # Keep only the last 5 checkpoints
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=10 if eval_dataset else None,  # Evaluate every 50 steps
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to=REPORT_TO,
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Use multiple workers for data loading
        resume_from_checkpoint=resume_from_checkpoint,  # Resume from checkpoint
    )

    # Define callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),  # Stop if no improvement for 3 evaluations
        CustomLoggingCallback(),  # Custom logging
    ]

    # Set up the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=callbacks,
    )
    return trainer

def run_inference(model, tokenizer, image, instruction):
    """Run inference on a single image."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=800,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )