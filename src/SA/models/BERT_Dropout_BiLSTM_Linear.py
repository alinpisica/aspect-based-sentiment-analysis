import torch

class BERT_Dropout_BiLSTM_Linear(torch.nn.Module):
    def __init__(self, bert, dropout, bilstm_in_features, no_out_labels, device='cpu'):
        super(BERT_Dropout_BiLSTM_Linear, self).__init__()          
        
        self.device = device

        self.bert = bert.to(device)

        self.dropout = torch.nn.Dropout(dropout)

        self.lstm = torch.nn.LSTM(self.bert.config.hidden_size, bilstm_in_features // 2, bidirectional=True, batch_first=True)
        
        self.fc = torch.nn.Linear(bilstm_in_features, no_out_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _, pooled_output= self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)

        dropout_output = self.dropout(pooled_output)

        lstm_output = self.lstm(dropout_output.unsqueeze(1))
        
        output = self.fc(lstm_output[0]).squeeze(1)

        return output