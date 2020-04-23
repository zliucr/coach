
from src.ner.datareader import datareader, PAD_INDEX, y1_set, y2_set
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import logging
logger = logging.getLogger()

entity_list = ["LOC", "PER", "ORG", "MISC"]

random_index = [491, 1937, 1465, 549, 871, 206, 1044, 1900, 159, 1695, 269, 1440, 570, 756, 1119, 595, 1540, 1868, 414, 772, 51, 1320, 1255, 1361, 192, 603, 1925, 1136, 336, 1873, 1885, 1271, 1796, 662, 1849, 1631, 1794, 780, 1725, 424, 325, 1750, 1163, 12, 1432, 872, 1830, 933, 56, 1704]

class Dataset(data.Dataset):
    def __init__(self, X, y1, y2, template_list=None):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.template_list = template_list

    def __getitem__(self, index):
        if self.template_list is None:
            return self.X[index], self.y1[index], self.y2[index]
        else:
            return self.X[index], self.y1[index], self.y2[index], self.template_list[index]
    
    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y1, y2 = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    
    return padded_seqs, lengths, y1, y2

def collate_fn_for_label_encoder(data):
    X, y1, y2, templates = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    
    tem_lengths = [len(sample_tem[0]) for sample_tem in templates]
    max_tem_len = max(tem_lengths)
    padded_templates = torch.LongTensor(len(templates), 3, max_tem_len).fill_(PAD_INDEX)
    for j, sample_tem in enumerate(templates):
        length = tem_lengths[j]
        padded_templates[j, 0, :length] = torch.LongTensor(sample_tem[0])
        padded_templates[j, 1, :length] = torch.LongTensor(sample_tem[1])
        padded_templates[j, 2, :length] = torch.LongTensor(sample_tem[2])
    tem_lengths = torch.LongTensor(tem_lengths)
    
    return padded_seqs, lengths, y1, y2, padded_templates, tem_lengths

def get_dataloader(batch_size, use_label_encoder=False, n_samples=0):
    data_train_bin, data_val_bin, data_test_bin, vocab = datareader(use_label_encoder)

    if n_samples != 0:
        sample_indices = random_index[:n_samples]
        size = len(data_test_bin["text"])
        cnt = 0
        for i in range(size):
            if i in sample_indices:
                cnt += 1
                data_train_bin["text"].append(data_test_bin["text"][i])
                data_train_bin["y1"].append(data_test_bin["y1"][i])
                data_train_bin["y2"].append(data_test_bin["y2"][i])
                data_test_bin["text"].pop(i)
                data_test_bin["y1"].pop(i)
                data_test_bin["y2"].pop(i)
                if use_label_encoder:
                    data_train_bin["template_list"].append(data_test_bin["template_list"][i])
                    data_test_bin["template_list"].pop(i)
        logger.info("few-shot learning on %d samples in the target domain" % cnt)

    if use_label_encoder:
        dataset_tr = Dataset(data_train_bin["text"], data_train_bin["y1"], data_train_bin["y2"], data_train_bin["template_list"])
    else:
        dataset_tr = Dataset(data_train_bin["text"], data_train_bin["y1"], data_train_bin["y2"])
    dataset_val = Dataset(data_val_bin["text"], data_val_bin["y1"], data_val_bin["y2"])
    dataset_test = Dataset(data_test_bin["text"], data_test_bin["y1"], data_test_bin["y2"])

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_label_encoder if use_label_encoder else collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test, vocab