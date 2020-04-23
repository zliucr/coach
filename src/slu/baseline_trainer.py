
from src.slu.datareader import domain_set, y1_set, y2_set
from preprocess.gen_embeddings_for_slu import domain2slot
import torch
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger()

from src.conll2002_metrics import *

O_INDEX = y1_set.index("O")
B_INDEX = y1_set.index("B")
I_INDEX = y1_set.index("I")

class BaselineTrainer(object):
    def __init__(self, params, slu_tagger):
        self.params = params
        self.slu_tagger = slu_tagger
        self.lr = params.lr
        self.optimizer = torch.optim.Adam(self.slu_tagger.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0
        self.stop_training_flag = False
    
    def train_step(self, X, lengths, y, y_dm):
        self.slu_tagger.train()

        predictions_for_batch = self.slu_tagger(y_dm, X)
        loss_list = []
        self.optimizer.zero_grad()
        for i, length in enumerate(lengths):
            predictions = predictions_for_batch[i][:,:length,:]
            golds = torch.LongTensor(y[i]).cuda()
            
            predictions = predictions.contiguous()
            golds = golds.contiguous()
            predictions = predictions.view(predictions.size()[0]*predictions.size()[1], 3)
            golds = golds.view(golds.size()[0]*golds.size()[1])
            
            loss = self.loss_fn(predictions, golds)
            loss.backward(retain_graph=True)
            
            loss_list.append(loss.item())

        self.optimizer.step()

        return np.mean(loss_list)
    
    def convert_slot_based_preds_to_original_preds(self, e, preds, dm_id):
        """
        Inputs:
            preds: preditions from baseline model (num_slot, seq_len)
            dm_id: domain id
        Outputs:
            final_predictions: final predictions (seq_len)
        """
        slot_list = domain2slot[domain_set[dm_id]]
        
        nonzero_pois = torch.nonzero(preds)   # a list of 2-d positions
        final_predictions = torch.LongTensor(preds.size()[1]).fill_(O_INDEX)
        for poi in nonzero_pois:
            slot_name_index = poi[0]
            length_poi = poi[1]
            pred = preds[poi[0]][poi[1]].item()
            slot_name = slot_list[slot_name_index]
            slot_name = "B-" + slot_name if pred == B_INDEX else "I-" + slot_name
            
            if slot_name in y2_set:
                # predicted slot name may not be in the y2_set, for example, I-party_size_number not in y2_set
                final_predictions[length_poi] = y2_set.index(slot_name)

        return final_predictions
    
    def convert_preds_to_binary_preds(self, preds):
        """
        Inputs:
            preds: (num_preds)
        Outputs:
            binary_preds: (num_preds)
        """
        binary_preds = []
        for pred_index in preds:
            binary_label = y2_set[pred_index.item()][0]
            binary_preds.append(y1_set.index(binary_label))
        
        return binary_preds

    def evaluate(self, e, dataloader, istestset=False):
        self.slu_tagger.eval()

        preds, golds = [], []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (X, lengths, y, y_dm) in pbar:
            X, lengths = X.cuda(), lengths.cuda()

            golds.extend(y)
            predictions_for_batch = self.slu_tagger(y_dm, X)
            # for slot_based_preds, dm_id in zip(predictions_for_batch, y_dm):
            for i, length in enumerate(lengths):
                dm_id = y_dm[i]
                slot_based_preds = predictions_for_batch[i][:,:length,:]

                slot_based_preds = torch.argmax(slot_based_preds, dim=-1) # convert (num_slot, seq_len, num_binslot) ==> (num_slot, seq_len)
                
                final_predictions = self.convert_slot_based_preds_to_original_preds(e, slot_based_preds, dm_id)
                preds.extend(final_predictions)
        
        # labels
        golds = np.concatenate(golds, axis=0)
        golds = list(golds)

        # final predictions
        preds = list(preds)

        # convert to binary
        bin_golds = self.convert_preds_to_binary_preds(golds)
        bin_preds = self.convert_preds_to_binary_preds(preds)

        lines = []
        bin_lines = []
        for pred, gold, bin_pred, bin_gold in zip(preds, golds, bin_preds, bin_golds):
            pred = pred.item()
            pred = y2_set[pred]
            gold = y2_set[gold]

            bin_pred = y1_set[bin_pred]
            bin_gold = y1_set[bin_gold]
            
            lines.append("w" + " " + pred + " " + gold)
            bin_lines.append("w" + " " + bin_pred + " " + bin_gold)
            
        result = conll2002_measure(lines)
        f1_score = result["fb1"]

        bin_result = conll2002_measure(bin_lines)
        bin_f1_score = bin_result["fb1"]

        if istestset == False:  # dev set
            if f1_score > self.best_f1:
                self.best_f1 = f1_score
                self.no_improvement_num = 0
                logger.info("Found better model!!")
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
            
            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True
        
        return bin_f1_score, f1_score, self.stop_training_flag

    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "slu_tagger": self.slu_tagger
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
