
from src.slu.datareader import datareader, y1_set, y2_set, domain_set, PAD_INDEX
from config import get_params
from preprocess.gen_embeddings_for_slu import domain2slot
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import logging
logger = logging.getLogger()

class Dataset(data.Dataset):
    def __init__(self, X, y, domains):
        self.X = X
        self.y = y
        self.domains = domains
    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domains[index]
    
    def __len__(self):
        return len(self.X)


def convert_labels_to_slot_based_labels(label_list, dm_name):
    slot_list = domain2slot[dm_name]
    new_labels = np.zeros((len(slot_list), len(label_list)))
    new_labels.fill(y1_set.index("O"))
    
    for i, label_index in enumerate(label_list):
        label_name = y2_set[label_index]
        splits = label_name.split("-")
        slot_type = splits[0]
        if slot_type == "O":
            continue
        else:
            slot_name = splits[1]
            slot_name_index = slot_list.index(slot_name)
            new_labels[slot_name_index, i] = y1_set.index(slot_type)
    
    return new_labels


def collate_fn(data):
    X, y, domains = zip(*data)
    lengths = [len(sample_x) for sample_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)

    return padded_seqs, lengths, y, domains


def get_dataloader(tgt_domain, batch_size, n_samples):
    all_data, vocab = datareader()
    train_data = {"utter": [], "y2": [], "new_y": [], "domains": []}
    
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["utter"].extend(dm_data["utter"])
            train_data["y2"].extend(dm_data["y2"])
            train_data["domains"].extend(dm_data["domains"])

    val_data = {"utter": [], "y2": [], "domains": []}
    test_data = {"utter": [], "y2": [], "domains": []}
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][:500]  
        val_data["y2"] = all_data[tgt_domain]["y2"][:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][:500]

        # the rest as test set
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]    
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]      # rest as test set
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]    # rest as test set

    else:
        # first n samples as train set
        train_data["utter"].extend(all_data[tgt_domain]["utter"][:n_samples])
        train_data["y2"].extend(all_data[tgt_domain]["y2"][:n_samples])
        train_data["domains"].extend(all_data[tgt_domain]["domains"][:n_samples])

        # from n to 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][n_samples:500]  
        val_data["y2"] = all_data[tgt_domain]["y2"][n_samples:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][n_samples:500]

        # the rest as test set (same as zero-shot)
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]

    # generate new labels
    for label_list, dm_id in zip(train_data["y2"], train_data["domains"]):
        dm_name = domain_set[dm_id]
        train_data["new_y"].append(convert_labels_to_slot_based_labels(label_list, dm_name))
    
    dataset_tr = Dataset(train_data["utter"], train_data["new_y"], train_data["domains"])
    dataset_val = Dataset(val_data["utter"], val_data["y2"], val_data["domains"])
    dataset_test = Dataset(test_data["utter"], test_data["y2"], test_data["domains"])

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test, vocab