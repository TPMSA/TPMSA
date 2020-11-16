import os
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from constants import DATA_PATH


class MMDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.vids = list(self.data.keys())
        self.transform = transform

    def __getitem__(self, idx):
        vid = self.vids[idx]
        sample = self.data[vid]
        return sample

    def __len__(self):
        return len(self.vids)


def collate_fn(batch_data):
    video = batch_data[0]
    sorted_data = sorted(video.items(), key=lambda item: int(item[0].split('[')[1].split(']')[0]))
    input_ids, t_masks, visual, va_masks, audio, labels = [], [], [], [], [], []
    for i in range(len(sorted_data)):
        t_feats, _, _, label = sorted_data[i][1]
        input_id = torch.tensor(tokenizer.encode(t_feats))
        t_mask = torch.ones_like(input_id, dtype=torch.float)
        input_ids.append(input_id)
        t_masks.append(t_mask)
        labels.append(torch.tensor(label))
    input_ids = pad_sequence(input_ids, batch_first=True)
    t_masks = pad_sequence(t_masks, batch_first=True)
    labels = torch.tensor(labels)
    return [input_ids, t_masks, labels]


def get_loader(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    dataset = MMDataset(data)
    loader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn)
    return loader


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader = get_loader(os.path.join(DATA_PATH, 'train.pkl'))
dev_loader = get_loader(os.path.join(DATA_PATH, 'dev.pkl'))
test_loader = get_loader(os.path.join(DATA_PATH, 'test.pkl'))
