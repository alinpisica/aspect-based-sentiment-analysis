import torch
import torch.nn as nn

class Embedding_BiLSTM_Linear(nn.Module):
    def __init__(self, embedding_model, emb_size, bilstm_size, no_out_labels, droput=0.3, device='cpu'):
        super().__init__()          
        
        self.device = device

        self.embedding = embedding_model

        self.dropout = nn.Dropout(droput)

        self.lstm = nn.LSTM(emb_size, bilstm_size // 2, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(bilstm_size, no_out_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _, embedded = self.embedding(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)

        output_2 = self.dropout(embedded.to(self.device)).to(self.device)

        output_3 = self.lstm(output_2.unsqueeze(1))
        
        fc_out = self.fc(output_3[0]).squeeze(1)

        return fc_out