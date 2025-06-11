import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_scheduler
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
pretrained_model = "uer/t5-base-chinese-cluecorpussmall"
max_source_seq_len = 256
max_target_seq_len = 32
batch_size = 16
num_train_epochs = 20
valid_steps = 200
logging_steps = 10
learning_rate = 5e-5
save_dir = "./checkpoints"

model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# 自定义数据集类
class QADataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = (
            f"问题：{item['question']}{tokenizer.sep_token}原文：{item['context']}"
        )
        output_seq = f"答案：{item['answer']}{tokenizer.eos_token}"
        return input_seq, output_seq
    


train_dataset = QADataset("./data/DuReaderQG/train.json")
print("train dataset size: ", len(train_dataset))
q, a = train_dataset[0]
