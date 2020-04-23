
from src.modules import Lstm
from src.utils import load_embedding_from_pkl
from src.slu.datareader import domain_set
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
            inputs: (num_slot, seq_len, hidden_dim+emb_dim)
        Outputs:
            predictions: (num_slot, seq_len, num_binslot)
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

        self.lstm_encoder = Lstm(params, vocab)
        self.lstm_predictor = LstmBasedSlotPredictor(params)
        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)
        self.example_embs = load_embedding_from_pkl(params.example_emb_file)

    def forward(self, domains, X):
        """
        Input:
            domains: domain list for each sample (bsz,)
            x: text (bsz, seq_len)
        Output:
            predictions: (bsz, num_binslot)
        """
        enc_hiddens = self.lstm_encoder(X)  # (bsz, seq_len, hidden_dim)

        predictions_for_batch = []
        bsz = domains.size()[0]
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            slot_embs_based_domain = torch.FloatTensor(self.slot_embs[domain_name]).cuda() # (slot_num, emb_dim)
            slot_embs_size = slot_embs_based_domain.size()

            hidden_i = enc_hiddens[i]  # (seq_len, hidden_dim)
            hidden_i_size = hidden_i.size()

            if self.use_example:
                example_embs_based_domain = torch.FloatTensor(self.example_embs[domain_name]).cuda() # (num_slot, hidden_dim, 2)
                embs_example1_based_domain = example_embs_based_domain[:, :, 0]  # (num_slot, hidden_dim)
                embs_example2_based_domain = example_embs_based_domain[:, :, 1]  # (num_slot, hidden_dim)

                temp_example1 = torch.matmul(hidden_i * self.w_a, embs_example1_based_domain.transpose(0,1)).unsqueeze(-1) # (seq_len, num_slot, 1)
                temp_example2 = torch.matmul(hidden_i * self.w_a, embs_example2_based_domain.transpose(0,1)).unsqueeze(-1) # (seq_len, num_slot, 1)
                example_softmax = self.softmax(torch.cat((temp_example1, temp_example2), dim=-1))  # (seq_len, num_slot, 2)

                example_attn_feature = example_softmax[:, :, 0].unsqueeze(-1) * embs_example1_based_domain + example_softmax[:, :, 1].unsqueeze(-1) * embs_example2_based_domain  # (seq_len, num_slot, hidden_dim)
                example_attn_feature = example_attn_feature.transpose(0, 1)  # (num_slot, seq_len, hidden_dim)

            ## combine hidden_i with slot_embs ==> (slot_num, seq_len, hidden_dim+emb_dim) ##
            # Step 1: expand hidden_i into (slot_num, seq_len, hidden_dim)
            hidden_i = hidden_i.unsqueeze(0)  # (1, seq_len, hidden_dim)
            hidden_i = hidden_i.expand(slot_embs_size[0], hidden_i_size[0], hidden_i_size[1])  # (slot_num, seq_len, hidden_dim)
            # Step 2: expand slot_embs_based_domain into (slot_num, seq_len, emb_dim)
            slot_embs_based_domain = slot_embs_based_domain.unsqueeze(1) # (slot_num, 1, emb_dim)
            slot_embs_based_domain = slot_embs_based_domain.expand(slot_embs_size[0], hidden_i_size[0], slot_embs_size[1])
            # Step3: torch.cat()
            combined_hidden_i = torch.cat((hidden_i, slot_embs_based_domain), dim=-1)  # (slot_num, seq_len, hidden_dim+emb_dim)

            if self.use_example:
                combined_hidden_i = torch.cat((combined_hidden_i, example_attn_feature), dim=-1)   # (slot_num, seq_len, hidden_dim+emb_dim+hidden_dim)

            pred = self.lstm_predictor(combined_hidden_i)
            predictions_for_batch.append(pred)

        return predictions_for_batch
