
from src.modules import Lstm, Attention, CRF
from src.utils import load_embedding_from_npy
from src.ner.datareader import ENTITY_PAD
import torch
from torch import nn
from torch.nn import functional as F

class LstmBasedSlotPredictor(nn.Module):
    def __init__(self, params):
        super(LstmBasedSlotPredictor, self).__init__()
        self.dropout = params.dropout
        self.bidirection = params.bidirection
        self.n_layer = params.n_layer
        self.num_binslot = params.num_binslot

        if params.use_example:
            self.input_dim = params.hidden_dim * 2 + params.emb_dim + params.hidden_dim * 2 if params.bidirection else params.hidden_dim + params.emb_dim + params.hidden_dim
        else:
            self.input_dim = params.hidden_dim * 2 + params.emb_dim if params.bidirection else params.hidden_dim + params.emb_dim
        
        self.hidden_dim = params.hidden_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, dropout=0.0, bidirectional=self.bidirection, batch_first=True)
        
        self.feature_dim = self.hidden_dim * 2 if params.bidirection else self.hidden_dim
        self.predictor = nn.Linear(self.feature_dim, self.num_binslot)
    
    def forward(self, inputs):
        """
        Inputs:
            inputs: (bsz, seq_len, hidden_dim+emb_dim)
        Outputs:
            predictions: (bsz, seq_len, num_binslot)
        """
        lstm_output, (_, _) = self.lstm(inputs)  # lstm_output: (num_slot, seq_len, hidden_dim)
        predictions = self.predictor(lstm_output)

        return predictions

class ConceptTagger(nn.Module):
    def __init__(self, params, vocab):
        super(ConceptTagger, self).__init__()
        
        self.use_example = params.use_example
        if self.use_example:
            hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
            self.w_a = nn.Parameter(torch.FloatTensor(hidden_dim))
            torch.nn.init.uniform(self.w_a.data, -0.01, 0.01)
            self.softmax = nn.Softmax(dim=-1)

            self.example_embs = torch.cuda.FloatTensor(load_embedding_from_npy(params.ner_example_emb_file))  # (num_entity, emb_dim, 2)

        self.lstm_encoder = Lstm(params, vocab)
        self.lstm_predictor = LstmBasedSlotPredictor(params)
        self.entity_embs = torch.cuda.FloatTensor(load_embedding_from_npy(params.ner_entity_type_emb_file))   # (num_entity, emb_dim)
        self.entity_embs_size = self.entity_embs.size()
        
    def forward(self, X):
        """
        Input:
            x: text (bsz, seq_len)
        Output:
            predictions: (bsz, seq_len, num_entity, num_binslot)
        """
        enc_hiddens = self.lstm_encoder(X)  # (bsz, seq_len, hidden_dim)
        
        if self.use_example:
            embs_example1 = self.example_embs[:, :, 0]  # (num_entity, emb_dim)
            embs_example2 = self.example_embs[:, :, 1]  # (num_entity, emb_dim)

            temp_example1 = torch.matmul(enc_hiddens * self.w_a, embs_example1.transpose(0,1)).unsqueeze(-1) # (bsz, seq_len, num_entity, 1)
            temp_example2 = torch.matmul(enc_hiddens * self.w_a, embs_example2.transpose(0,1)).unsqueeze(-1) # (bsz, seq_len, num_entity, 1)
            example_softmax = self.softmax(torch.cat((temp_example1, temp_example2), dim=-1))  # (bsz, seq_len, num_entity, 2)
            
            softmax_size = example_softmax.size()
            example_size = embs_example1.size()
            embs_example1 = embs_example1.unsqueeze(0).unsqueeze(1).expand(softmax_size[0], softmax_size[1], example_size[0], example_size[1])
            embs_example2 = embs_example2.unsqueeze(0).unsqueeze(1).expand(softmax_size[0], softmax_size[1], example_size[0], example_size[1])

            example_attn_feature = example_softmax[:, :, :, 0].unsqueeze(-1) * embs_example1 + example_softmax[:, :, :, 1].unsqueeze(-1) * embs_example2  # (bsz, seq_len, num_entity, emb_dim)
        
        enc_hiddens_size = enc_hiddens.size()
        enc_hiddens = enc_hiddens.unsqueeze(2).expand(enc_hiddens_size[0], enc_hiddens_size[1], self.entity_embs_size[0], enc_hiddens_size[2])  # (bsz, seq_len, num_entity, hidden_dim)
        
        entity_embs = self.entity_embs.unsqueeze(0).unsqueeze(1).expand(enc_hiddens_size[0], enc_hiddens_size[1], self.entity_embs_size[0], self.entity_embs_size[1]) # (bsz, seq_len, num_entity, emb_dim)

        combined_hidden = torch.cat((enc_hiddens, entity_embs), dim=-1) # (bsz, seq_len, num_entity, hidden_dim+emb_dim)

        if self.use_example:
            combined_hidden = torch.cat((combined_hidden, example_attn_feature), dim=-1) # (bsz, seq_len, num_entity, hidden_dim+emb_dim*2)

        # print(combined_hidden.size())
        slot_predictions_each_entity = []
        for i in range(self.entity_embs_size[0]):
            entity_i_hidden = combined_hidden[:,:,i,:]
            pred = self.lstm_predictor(entity_i_hidden)
            slot_predictions_each_entity.append(pred)
        
        slot_predictions_each_entity = torch.stack(slot_predictions_each_entity, dim=2)  # (bsz, seq_len, num_entity, num_binslot)

        return slot_predictions_each_entity


class BiLSTMCRFTagger(nn.Module):
    def __init__(self, params, vocab):
        super(BiLSTMCRFTagger, self).__init__()
        self.lstm = Lstm(params, vocab)
        self.num_entity_label = params.num_entity_label
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_entity_label)

        self.crf_layer = CRF(self.num_entity_label)
        
    def forward(self, X, lengths=None):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_binslot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        return prediction
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(ENTITY_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y
