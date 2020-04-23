
from src.ner.datareader import y1_set, y2_set
from src.ner.dataloader import entity_list
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger()

from src.conll2002_metrics import *

class NERTrainer(object):
    def __init__(self, params, binary_ner_tagger, entityname_predictor, sent_repre_generator=None):
        self.binary_ner_tagger = binary_ner_tagger
        self.entityname_predictor = entityname_predictor
        self.lr = params.lr
        self.num_domain = params.num_domain
        self.use_label_encoder = params.tr

        if self.use_label_encoder:
            self.sent_repre_generator = sent_repre_generator
            self.loss_fn_mse = nn.MSELoss()
            model_parameters = [
                {"params": self.binary_ner_tagger.parameters()},
                {"params": self.entityname_predictor.parameters()},
                {"params": self.sent_repre_generator.parameters()}
            ]
        else:
            model_parameters = [
                {"params": self.binary_ner_tagger.parameters()},
                {"params": self.entityname_predictor.parameters()}
            ]
        # Adam optimizer
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0

        self.stop_training_flag = False
    
    def train_step(self, X, lengths, y_bin, y_final, templates=None, tem_lengths=None, epoch=None):
        self.binary_ner_tagger.train()
        self.entityname_predictor.train()
        if self.use_label_encoder:
            self.sent_repre_generator.train()

        bin_preds, lstm_hiddens = self.binary_ner_tagger(X)

        ## optimize binary_ner_tagger
        loss_bin = self.binary_ner_tagger.crf_loss(bin_preds, lengths, y_bin)
        self.optimizer.zero_grad()
        loss_bin.backward(retain_graph=True)
        self.optimizer.step()

        ## optimize entityname_predictor
        pred_entityname_list, gold_entityname_list = self.entityname_predictor(lstm_hiddens, binary_golds=y_bin, final_golds=y_final)

        for pred_entityname_each_sample, gold_entityname_each_sample in zip(pred_entityname_list, gold_entityname_list):
            if pred_entityname_each_sample is None:
                continue
            assert pred_entityname_each_sample.size()[0] == gold_entityname_each_sample.size()[0]

            loss_entityname = self.loss_fn(pred_entityname_each_sample, gold_entityname_each_sample.cuda())
            self.optimizer.zero_grad()
            loss_entityname.backward(retain_graph=True)
            self.optimizer.step()
        
        if self.use_label_encoder:
            templates_repre, input_repre = self.sent_repre_generator(templates, tem_lengths, lstm_hiddens, lengths)

            input_repre = input_repre.detach()
            template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
            template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
            template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
            input_repre.requires_grad = True
            
            # torch.nn.utils.clip_grad_norm_(self.binary_ner_tagger.parameters(), 20)
            # torch.nn.utils.clip_grad_norm_(self.entityname_predictor.parameters(), 20)
            # torch.nn.utils.clip_grad_norm_(self.sent_repre_generator.parameters(), 20)
            self.optimizer.zero_grad()
            template0_loss.backward(retain_graph=True)
            template1_loss.backward(retain_graph=True)
            template2_loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch > 3:
                templates_repre = templates_repre.detach()
                input_loss0 = self.loss_fn_mse(input_repre, templates_repre[:, 0, :])
                input_loss1 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 1, :])
                input_loss2 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 2, :])
                templates_repre.requires_grad = True

                self.optimizer.zero_grad()
                input_loss0.backward(retain_graph=True)
                input_loss1.backward(retain_graph=True)
                input_loss2.backward(retain_graph=True)
                self.optimizer.step()

        if self.use_label_encoder:
            return loss_bin.item(), loss_entityname.item(), template0_loss.item(), template1_loss.item()
        else:
            return loss_bin.item(), loss_entityname.item()
    
    def evaluate(self, dataloader, istestset=False):
        self.binary_ner_tagger.eval()
        self.entityname_predictor.eval()

        binary_preds, binary_golds = [], []
        final_preds, final_golds = [], []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (X, lengths, y_bin, y_final) in pbar:
            binary_golds.extend(y_bin)
            final_golds.extend(y_final)

            X, lengths = X.cuda(), lengths.cuda()
            bin_preds_batch, lstm_hiddens = self.binary_ner_tagger(X)
            bin_preds_batch = self.binary_ner_tagger.crf_decode(bin_preds_batch, lengths)
            binary_preds.extend(bin_preds_batch)

            entityname_preds_batch = self.entityname_predictor(lstm_hiddens, binary_predictions=bin_preds_batch, binary_golds=None, final_golds=None)
            
            final_preds_batch = self.combine_binary_and_entityname_preds(bin_preds_batch, entityname_preds_batch)
            final_preds.extend(final_preds_batch)
        
        # binary predictions
        binary_preds = np.concatenate(binary_preds, axis=0)
        binary_preds = list(binary_preds)
        binary_golds = np.concatenate(binary_golds, axis=0)
        binary_golds = list(binary_golds)

        # final predictions
        final_preds = np.concatenate(final_preds, axis=0)
        final_preds = list(final_preds)
        final_golds = np.concatenate(final_golds, axis=0)
        final_golds = list(final_golds)

        bin_lines, final_lines = [], []
        for bin_pred, bin_gold, final_pred, final_gold in zip(binary_preds, binary_golds, final_preds, final_golds):
            bin_entity_pred = y1_set[bin_pred]
            bin_entity_gold = y1_set[bin_gold]
            
            final_entity_pred = y2_set[final_pred]
            final_entity_gold = y2_set[final_gold]
            
            bin_lines.append("w" + " " + bin_entity_pred + " " + bin_entity_gold)
            final_lines.append("w" + " " + final_entity_pred + " " + final_entity_gold)
            
        bin_result = conll2002_measure(bin_lines)
        bin_f1 = bin_result["fb1"]
        
        final_result = conll2002_measure(final_lines)
        final_f1 = final_result["fb1"]
        
        if istestset == False:  # dev set
            if final_f1 > self.best_f1:
                self.best_f1 = final_f1
                self.no_improvement_num = 0
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
            
            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True
        
        return bin_f1, final_f1, self.stop_training_flag
        
    def combine_binary_and_entityname_preds(self, binary_preds_batch, entityname_preds_batch):
        """
        Input:
            binary_preds: (bsz, seq_len)
            entityname_preds: (bsz, num_entityname, entity_num)
        Output:
            final_preds: (bsz, seq_len)
        """
        final_preds = []
        for i in range(len(binary_preds_batch)):
            binary_preds = binary_preds_batch[i]
            entityname_preds = entityname_preds_batch[i]
            
            i = -1
            final_preds_each = []
            for bin_pred in binary_preds:
                # values of bin_pred are 0 (O), or 1(B) or 2(I)
                if bin_pred.item() == 0:
                    final_preds_each.append(0)
                elif bin_pred.item() == 1:
                    i += 1
                    pred_entity_id = torch.argmax(entityname_preds[i])
                    entityname = "B-" + entity_list[pred_entity_id]
                    final_preds_each.append(y2_set.index(entityname))
                elif bin_pred.item() == 2:
                    if i == -1:
                        final_preds_each.append(0)
                    else:
                        pred_entity_id = torch.argmax(entityname_preds[i])
                        entityname = "I-" + entity_list[pred_entity_id]
                        if entityname not in y2_set:
                            final_preds_each.append(0)
                        else:
                            final_preds_each.append(y2_set.index(entityname))
                
            assert len(final_preds_each) == len(binary_preds)
            final_preds.append(final_preds_each)

        return final_preds
