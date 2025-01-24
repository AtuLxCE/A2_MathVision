# models/model_loader.py

from unsloth import FastVisionModel
import torch
from config.settings import MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer

def configure_lora(model):
    """Configure LoRA for the model."""
    from config.settings import (
        LORA_R,
        LORA_ALPHA,
        LORA_DROPOUT,
        LORA_BIAS,
        LORA_RANDOM_STATE,
        LORA_USE_RSLORA,
        LORA_LOFTQ_CONFIG,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        random_state=LORA_RANDOM_STATE,
        use_rslora=LORA_USE_RSLORA,
        loftq_config=LORA_LOFTQ_CONFIG,
    )
    return model