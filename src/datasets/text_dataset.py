import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

class TextDataset:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = self.prepare_data()
        self.input_size = len(self.tokenizer.vocab)
        self.output_size = len(self.tokenizer.vocab)

    def prepare_data(self):
        dataset = load_dataset(self.config['type'], self.config['name'], split=self.config['split'])

        def tokenize(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.config['max_length'])

        tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
        tokenized_dataset.set_format(type='torch', columns=['input_ids'])
        return tokenized_dataset

    def get_data(self):
        return self.dataset, self.input_size, self.output_size

    def get_criterion(self):
        return nn.CrossEntropyLoss()
