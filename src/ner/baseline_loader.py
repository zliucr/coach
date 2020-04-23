
from src.ner.datareader import datareader, PAD_INDEX, y1_set, y2_set
from src.ner.dataloader import random_index
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import logging
logger = logging.getLogger()

entity_list = ["LOC", "PER", "ORG", "MISC"]

class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)


def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    
    return padded_seqs, lengths, y

def convert_labels_to_slot_based_labels(label_list):
    new_labels = np.zeros((len(entity_list), len(label_list)))
    new_labels.fill(y1_set.index("O"))
    
    for i, label_index in enumerate(label_list):
        label_name = y2_set[label_index]
        splits = label_name.split("-")
        slot_type = splits[0]
        if slot_type == "O":
            continue
        else:
            slot_name = splits[1]
            slot_name_index = entity_list.index(slot_name)
            new_labels[slot_name_index, i] = y1_set.index(slot_type+"-Entity")
    
    return new_labels

def get_dataloader(batch_size, bilstmcrf=False, n_samples=0):
    data_train_bin, data_val_bin, data_test_bin, vocab = datareader()

    if n_samples != 0:
        sample_indices = random_index[:n_samples]
        size = len(data_test_bin["text"])
        cnt = 0
        for i in range(size):
            if i in sample_indices:
                cnt += 1
                data_train_bin["text"].append(data_test_bin["text"][i])
                data_train_bin["y2"].append(data_test_bin["y2"][i])
                data_test_bin["text"].pop(i)
                data_test_bin["y2"].pop(i)
        logger.info("few-shot learning on %d samples in the target domain" % cnt)

    if bilstmcrf:
        dataset_tr = Dataset(data_train_bin["text"], data_train_bin["y2"])
    else:
        data_train_bin["new_y"] = []
        # generate new labels
        for label_list in data_train_bin["y2"]:
            data_train_bin["new_y"].append(convert_labels_to_slot_based_labels(label_list))
        dataset_tr = Dataset(data_train_bin["text"], data_train_bin["new_y"])
    dataset_val = Dataset(data_val_bin["text"], data_val_bin["y2"])
    dataset_test = Dataset(data_test_bin["text"], data_test_bin["y2"])

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test, vocab