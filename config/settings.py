# config/settings.py

# Hugging Face token
HF_TOKEN = ""

# Model and training settings
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect
LOAD_IN_4BIT = True

# LoRA settings
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_BIAS = "none"
LORA_RANDOM_STATE = 3407
LORA_USE_RSLORA = False
LORA_LOFTQ_CONFIG = None

# Training settings
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 5
MAX_STEPS = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
SEED = 3407
OUTPUT_DIR = "output/"
REPORT_TO = "wandb"

# Instruction for the model
INSTRUCTION = "You are an expert geometric problem solver. Provide a detailed description of the image, ensuring that all necessary information is included to solve any related questions without requiring further clarification."

# W&B settings
WANDB_PROJECT = "A2-ImageModel-FineTuning"  # W&B project name
WANDB_RUN_ID = None  # Set to a specific run ID to resume
WANDB_RESUME = "allow"  # Allow resuming runs


# Hugging Face Hub settings
HUGGINGFACE_HUB_REPO_NAME = "A2_ImageModel_200Epoch"  # Your repository name
HUGGINGFACE_HUB_TOKEN = "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"  # Your Hugging Face token