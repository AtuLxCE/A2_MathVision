# data/dataset.py

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

def load_image_dataset():
    """Load the image dataset."""
    return load_dataset("Yugratna/geometric_image_description_data", split="train")

def convert_to_conversation(sample, instruction):
    """Convert a dataset sample into a conversation format."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["dominated_image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text_only_question"]}],
        },
    ]
    return {"messages": conversation}

def split_dataset(dataset, test_size=0.2, random_state=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        dataset: The dataset to split.
        test_size: The proportion of the dataset to include in the validation split.
        random_state: Random seed for reproducibility.
    
    Returns:
        train_data: Training dataset.
        val_data: Validation dataset.
    """
    # Convert the dataset to a list of samples
    samples = [sample for sample in dataset]

    # Split the dataset into train and validation sets
    train_samples, val_samples = train_test_split(
        samples, test_size=test_size, random_state=random_state
    )

    # Convert back to Dataset objects
    train_data = Dataset.from_list(train_samples)
    val_data = Dataset.from_list(val_samples)

    return train_data, val_data