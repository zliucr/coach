
import numpy as np
import logging
logger = logging.getLogger()

y1_set = ["O", "B-Entity", "I-Entity"]
y2_set = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

entity_label_to_descrip = {
    "LOC": "location", 
    "PER": "person", 
    "ORG": "organization", 
    "MISC": "miscellaneous"}
entity_types = ["location", "person", "organization", "miscellaneous"]

ENTITY_PAD = 0
PAD_INDEX = 0
UNK_INDEX = 1


class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX, "UNK":UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2
    def index_words(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1

def read_file(filepath, vocab, use_label_encoder=False):
    text_list = []
    label_list1 = []
    label_list2 = []
    if use_label_encoder:
        template_list = []
    with open(filepath, "r") as f:
        tok_list, l1_list, l2_list = [], [], []
        for i, line in enumerate(f):
            line = line.strip()
            if line == "":
                if len(tok_list) > 0:
                    # update vocab
                    vocab.index_words(tok_list)

                    if use_label_encoder:
                        # remove the training samples with only O
                        filtered = True
                        for l2 in l2_list:
                            if l2 != "O": filtered = False
                        if filtered == True:
                            tok_list, l1_list, l2_list = [], [], []
                            continue
                    assert len(tok_list) == len(l1_list) == len(l2_list)
                    text_list.append(tok_list)
                    label_list1.append(l1_list)
                    label_list2.append(l2_list)

                    if use_label_encoder:
                        template_each_sample = [[],[],[]]
                        # generate template labels
                        for token, l2 in zip(tok_list, l2_list):
                            if l2[0] == "I": continue
                            if l2 == "O":
                                template_each_sample[0].append(token)
                                template_each_sample[1].append(token)
                                template_each_sample[2].append(token)
                            else:
                                # "B" in l2
                                entity_name = l2.split("-")[1]
                                template_each_sample[0].append(entity_label_to_descrip[entity_name])
                                idx = 0
                                for j in range(1, 3):  # j from 1 to 2
                                    if entity_types[idx] != entity_label_to_descrip[entity_name]:
                                        template_each_sample[j].append(entity_types[idx])
                                    else:
                                        idx = idx + 1
                                        template_each_sample[j].append(entity_types[idx])
                                    idx = idx + 1

                        template_list.append(template_each_sample)

                tok_list, l1_list, l2_list = [], [], []
                continue
            splits = line.split()
            tok_list.append(splits[0].lower())
            l1_list.append(splits[1])
            l2_list.append(splits[2])
    
    if use_label_encoder:
        data_dict = {"text": text_list, "y1": label_list1, "y2": label_list2, "template_list": template_list}
    else:
        data_dict = {"text": text_list, "y1": label_list1, "y2": label_list2}

    return data_dict, vocab

def binarize_data(data, vocab, use_label_encoder=False):
    if use_label_encoder:
        data_bin = {"text": [], "y1": [], "y2": [], "template_list": []}
    else:
        data_bin = {"text": [], "y1": [], "y2": []}

    for tokens, y1_list, y2_list in zip(data["text"], data["y1"], data["y2"]):
        text_bin, y1_bin, y2_bin = [], [], []
        # binarize text
        for token in tokens:
            text_bin.append(vocab.word2index[token])
        data_bin["text"].append(text_bin)
        # binarize y1
        for y1 in y1_list:
            y1_bin.append(y1_set.index(y1))
        data_bin["y1"].append(y1_bin)
        # binarize y2
        for y2 in y2_list:
            y2_bin.append(y2_set.index(y2))
        data_bin["y2"].append(y2_bin)
        assert len(text_bin) == len(y1_bin) == len(y2_bin)

    if use_label_encoder:
        for template_each_sample in data["template_list"]:
            template_each_sample_bin = [[],[],[]]
            
            for tok1, tok2, tok3 in zip(template_each_sample[0], template_each_sample[1], template_each_sample[2]):
                template_each_sample_bin[0].append(vocab.word2index[tok1])
                template_each_sample_bin[1].append(vocab.word2index[tok2])
                template_each_sample_bin[2].append(vocab.word2index[tok3])

            data_bin["template_list"].append(template_each_sample_bin)

    return data_bin

def datareader(use_label_encoder=False):
    logger.info("Loading and processing data ...")

    vocab = Vocab()
    # conll 2003 train and dev set
    # use_label_encoder only for training
    data_train, vocab = read_file("./data/ner/conll2003/train.txt", vocab, use_label_encoder)
    data_val, vocab = read_file("./data/ner/conll2003/dev.txt", vocab)

    # tech domain test set, use_label_encoder for few-shot learning
    data_test, vocab = read_file("./data/ner/tech/tech_test.txt", vocab, use_label_encoder)

    # binarize train data
    data_train_bin = binarize_data(data_train, vocab, use_label_encoder)
    # binarize dev data
    data_val_bin = binarize_data(data_val, vocab)
    # binarize test data
    data_test_bin = binarize_data(data_test, vocab, use_label_encoder)

    return data_train_bin, data_val_bin, data_test_bin, vocab
