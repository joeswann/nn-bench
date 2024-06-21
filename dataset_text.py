import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

class TextDataset:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = self.prepare_data()
        self.input_size = len(self.tokenizer.vocab)
        self.output_size = len(self.tokenizer.vocab)

    def prepare_data(self):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        def tokenize(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
        tokenized_dataset.set_format(type='torch', columns=['input_ids'])
        return tokenized_dataset

    def get_data(self):
        return self.dataset, self.input_size, self.output_size

    def get_hyperparams(self):
        return {
            'steps': 10,
            'step_size': 0.005,
            'solver': 'RungeKutta',
            'adaptive': True
        }

    def get_criterion(self):
        return nn.CrossEntropyLoss()
