
from src.ner.datareader import y1_set, y2_set
from src.ner.baseline_loader import entity_list
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger()

from src.conll2002_metrics import *

O_INDEX = y1_set.index("O")
B_INDEX = y1_set.index("B-Entity")
I_INDEX = y1_set.index("I-Entity")

class BaselineTrainer(object):
    def __init__(self, params, ner_tagger):
        self.ner_tagger = ner_tagger
        self.lr = params.lr
        self.optimizer = torch.optim.Adam(self.ner_tagger.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0
        self.stop_training_flag = False
    
    def train_step(self, X, lengths, y):
        self.ner_tagger.train()

        predictions_for_batch = self.ner_tagger(X)  # (bsz, seq_len, num_entity, num_binslot)
        loss_list = []
        self.optimizer.zero_grad()
        for i, length in enumerate(lengths):
            predictions = predictions_for_batch[i,:length,:,:]  # (seq_len, num_entity, num_binslot)
            predictions = predictions.transpose(0,1)  # (num_entity, seq_len, num_binslot)
            golds = torch.LongTensor(y[i]).cuda()  # (num_entity, seq_len)
            
            predictions = predictions.contiguous()
            golds = golds.contiguous()
            predictions = predictions.view(predictions.size()[0]*predictions.size()[1], 3)
            golds = golds.view(golds.size()[0]*golds.size()[1])
            
            loss = self.loss_fn(predictions, golds)
            loss.backward(retain_graph=True)
            
            loss_list.append(loss.item())

        self.optimizer.step()

        return np.mean(loss_list)
    
    def convert_entity_based_preds_to_original_preds(self, preds):
        """
        Inputs:
            preds: preditions from baseline model (num_entity, seq_len)
        Outputs:
            final_predictions: final predictions (seq_len)
        """
        nonzero_pois = torch.nonzero(preds)   # a list of 2-d positions
        final_predictions = torch.LongTensor(preds.size()[1]).fill_(O_INDEX)
        for poi in nonzero_pois:
            entity_name_index = poi[0]
            length_poi = poi[1]
            pred = preds[poi[0]][poi[1]].item()
            entity_name = entity_list[entity_name_index]
            entity_name = "B-" + entity_name if pred == B_INDEX else "I-" + entity_name
            
            final_predictions[length_poi] = y2_set.index(entity_name)

        return final_predictions

    def evaluate(self, dataloader, istestset=False):
        self.ner_tagger.eval()

        preds, golds = [], []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (X, lengths, y) in pbar:
            X, lengths = X.cuda(), lengths.cuda()

            golds.extend(y)
            predictions_for_batch = self.ner_tagger(X)
            for i, length in enumerate(lengths):
                entity_based_preds = predictions_for_batch[i,:length,:,:]  # (seq_len, num_entity, num_binslot)
                entity_based_preds = entity_based_preds.transpose(0,1) # (num_entity, seq_len, num_binslot)
                entity_based_preds = torch.argmax(entity_based_preds, dim=-1) # convert (num_entity, seq_len, num_binslot) ==> (num_entity, seq_len)
                
                final_predictions = self.convert_entity_based_preds_to_original_preds(entity_based_preds)
                preds.extend(final_predictions)
        
        # labels
        golds = np.concatenate(golds, axis=0)
        golds = list(golds)

        # final predictions
        preds = list(preds)

        lines = []
        for pred, gold in zip(preds, golds):
            pred = pred.item()
            pred = y2_set[pred]
            gold = y2_set[gold]
            
            lines.append("w" + " " + pred + " " + gold)
            
        result = conll2002_measure(lines)
        f1_score = result["fb1"]

        if istestset == False:  # dev set
            if f1_score > self.best_f1:
                self.best_f1 = f1_score
                self.no_improvement_num = 0
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
            
            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True
        
        return f1_score, self.stop_training_flag


class BiLSTMCRFTrainer(object):
    def __init__(self, params, ner_tagger):
        self.ner_tagger = ner_tagger
        self.lr = params.lr
        self.optimizer = torch.optim.Adam(self.ner_tagger.parameters(), lr=self.lr)

        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0

        self.stop_training_flag = False

    def train_step(self, X, lengths, y):
        self.ner_tagger.train()

        preds = self.ner_tagger(X)

        ## optimize ner_tagger
        loss = self.ner_tagger.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def evaluate(self, dataloader, istestset=False):
        self.ner_tagger.eval()

        preds, golds = [], []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (X, lengths, y) in pbar:
            golds.extend(y)

            X, lengths = X.cuda(), lengths.cuda()
            preds_batch = self.ner_tagger(X)
            preds_batch = self.ner_tagger.crf_decode(preds_batch, lengths)
            preds.extend(preds_batch)
        
        preds = np.concatenate(preds, axis=0)
        preds = list(preds)
        golds = np.concatenate(golds, axis=0)
        golds = list(golds)

        lines = []
        for pred, gold in zip(preds, golds):
            slot_pred = y2_set[pred]
            slot_gold = y2_set[gold]
            
            lines.append("w" + " " + slot_pred + " " + slot_gold)
            
        result = conll2002_measure(lines)
        f1 = result["fb1"]
        
        if istestset == False:  # dev set
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.no_improvement_num = 0
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
            
            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True
        
        return f1, self.stop_training_flag
    