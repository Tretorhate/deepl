"""
Medical Text Dataset Loading Module
Handles loading and preprocessing of medical text datasets (MedQuAD, PubMed, etc.)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re


class Tokenizer:
    """Simple word-level tokenizer for medical texts."""

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        for text in texts:
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)

        # Add most common words
        idx = 2
        for word, _ in self.word_freq.most_common(self.vocab_size - 2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

    def encode(self, text, max_length=128, padding='max_length', truncation=True):
        """Encode text to token indices."""
        tokens = self._tokenize(text)

        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]

        token_ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

        if padding == 'max_length':
            pad_length = max_length - len(token_ids)
            token_ids = token_ids + [self.word2idx['<PAD>']] * pad_length

        return torch.tensor(token_ids, dtype=torch.long)

    @staticmethod
    def _tokenize(text):
        """Basic tokenization: lowercase and split on whitespace/punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def get_vocab_size(self):
        """Return vocabulary size."""
        return len(self.word2idx)


class TextDataset(Dataset):
    """Medical text classification dataset."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of text strings
            labels: List of labels (int)
            tokenizer: Tokenizer object
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode text
        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.word2idx['<PAD>']).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_medquad_dataset(subset_size=None):
    """
    Load MedQuAD dataset (Medical Question-Answer Dataset).

    For full dataset, see: https://github.com/abachaa/MedQuAD
    For demo, we'll create synthetic data.

    Args:
        subset_size: Number of samples to use (None for full)

    Returns:
        (texts, labels, category_names) tuple
    """
    # Medical question samples for different categories
    medical_samples = {
        'Medications': [
            "What are the side effects of aspirin? Aspirin can cause bleeding and stomach ulcers.",
            "How should I take ibuprofen? Take with food to prevent stomach upset.",
            "What are the contraindications for metformin? Avoid in kidney disease.",
            "How does penicillin work? It disrupts bacterial cell wall synthesis.",
            "What is the dosage for amoxicillin? Typically 500mg three times daily.",
        ],
        'Procedures': [
            "What is a colonoscopy used for? To screen for colorectal cancer.",
            "How is an ultrasound performed? Sound waves create images of organs.",
            "What happens during an MRI scan? Strong magnets create detailed images.",
            "What is the purpose of a biopsy? To examine tissue for disease.",
            "How long does open heart surgery take? Usually 3-6 hours.",
        ],
        'Diseases': [
            "What are the symptoms of diabetes? Increased thirst and urination.",
            "How is hypertension treated? With lifestyle changes and medication.",
            "What causes heart disease? Plaque buildup in arteries.",
            "How do you diagnose asthma? Through breathing tests and symptoms.",
            "What are risk factors for cancer? Age, genetics, and lifestyle.",
        ],
        'Wellness': [
            "How much exercise do I need? At least 150 minutes per week.",
            "What is a healthy diet? Rich in vegetables, whole grains, and lean protein.",
            "How important is sleep? Adults need 7-9 hours daily.",
            "How can I reduce stress? Try meditation and exercise.",
            "What supplements are important? Vitamin D and B12 for many people.",
        ]
    }

    texts = []
    labels = []
    category_names = list(medical_samples.keys())
    category2idx = {cat: i for i, cat in enumerate(category_names)}

    # Create dataset
    for category, samples in medical_samples.items():
        label = category2idx[category]
        for sample in samples:
            texts.append(sample)
            labels.append(label)

    # Duplicate to create larger dataset if needed
    if subset_size is None:
        subset_size = len(texts) * 5

    original_len = len(texts)
    while len(texts) < subset_size:
        idx = np.random.randint(0, original_len)
        texts.append(texts[idx])
        labels.append(labels[idx])

    return texts, labels, category_names


def load_text_dataset(dataset_name, config):
    """
    Load a medical text dataset and create train/val/test splits.

    Args:
        dataset_name: Name of dataset ('MedQuAD', 'PubMed', etc.)
        config: Configuration dictionary with:
            - max_vocab_size
            - max_seq_length
            - batch_size (optional)

    Returns:
        Dictionary with:
        {
            'train': DataLoader for training
            'val': DataLoader for validation
            'test': DataLoader for testing
            'tokenizer': Fitted tokenizer
            'vocab_size': Size of vocabulary
            'num_classes': Number of output classes
            'class_names': Name of each class
        }
    """
    print(f"\n  Loading {dataset_name} dataset...")

    # Load raw data
    if 'MedQuAD' in dataset_name:
        texts, labels, class_names = load_medquad_dataset()
    else:
        print(f"  Warning: {dataset_name} not fully implemented, using MedQuAD")
        texts, labels, class_names = load_medquad_dataset()

    print(f"    Total samples: {len(texts)}")
    print(f"    Classes: {class_names}")

    # Create tokenizer
    tokenizer = Tokenizer(vocab_size=config['max_vocab_size'])
    tokenizer.build_vocab(texts)

    print(f"    Vocabulary size: {tokenizer.get_vocab_size()}")

    # Convert labels to numpy for splitting
    labels_array = np.array(labels)

    # Split into train (70%), val (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels_array, test_size=0.30, random_state=42, stratify=labels_array
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"    Train samples: {len(X_train)}")
    print(f"    Val samples: {len(X_val)}")
    print(f"    Test samples: {len(X_test)}")

    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer, config['max_seq_length'])
    val_dataset = TextDataset(X_val, y_val, tokenizer, config['max_seq_length'])
    test_dataset = TextDataset(X_test, y_test, tokenizer, config['max_seq_length'])

    # Create dataloaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Prepare output
    result = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'tokenizer': tokenizer,
        'vocab_size': tokenizer.get_vocab_size(),
        'num_classes': len(class_names),
        'class_names': class_names,
    }

    print(f"  Dataset loaded successfully!")

    return result


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


if __name__ == '__main__':
    # Test the data loader
    config = {
        'max_vocab_size': 5000,
        'max_seq_length': 128,
        'batch_size': 32,
    }

    data = load_text_dataset('MedQuAD', config)

    print("\nDataset loaded successfully!")
    print(f"Vocab size: {data['vocab_size']}")
    print(f"Num classes: {data['num_classes']}")
    print(f"Classes: {data['class_names']}")

    # Test batch loading
    batch = next(iter(data['train']))
    print(f"\nBatch sample:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  label shape: {batch['label'].shape}")
