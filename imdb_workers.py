import torch
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        input_ids = self.tokenizer(
            row["text"],
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]
        return {"input_ids": input_ids, "label": row["label"]}

class Collator:
    def __init__(self, pad_token_id):
        self.pad_id = pad_token_id

    def __call__(self, batch):
        sequences = [row["input_ids"] for row in batch]
        labels = [row["label"] for row in batch]
        max_len = max(map(len, sequences))

        ids = [seq + [self.pad_id] * (max_len - len(seq)) for seq in sequences]
        masks = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
