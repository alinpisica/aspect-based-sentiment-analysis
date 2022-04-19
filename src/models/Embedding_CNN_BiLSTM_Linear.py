import torch
import torch.nn as nn

class Embedding_CNN_BiLSTM_Linear(nn.Module):
    def __init__(self, embedding_model, emb_size, emb_seq_len, bilstm_size, no_out_labels,  
                conv_out_channels, conv_kernel_size, dropout=0.3, conv_dilation=1, conv_padding=0, conv_stride=1, device='cpu'):
        super().__init__()          
        
        self.device = device

        self.embedding = embedding_model

        self.dropout = nn.Dropout(dropout)

        self.convlayer = nn.Conv1d(emb_size, conv_out_channels, conv_kernel_size, dilation=conv_dilation, padding=conv_padding)

        conv_l_out_size = (emb_seq_len + 2 * conv_padding - conv_dilation * (conv_kernel_size - 1) - 1) // conv_stride + 1

        self.lstm = nn.LSTM(conv_l_out_size, bilstm_size // 2, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(bilstm_size * conv_out_channels, no_out_labels)
    
    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, _ = self.embedding(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        embedded = self.dropout(last_hidden_state.to(self.device)).to(self.device).permute(0, 2, 1)
        
        conv_output = self.convlayer(embedded).to(self.device)

        lstm_output = self.lstm(conv_output)
        
        fc_out = self.fc(lstm_output[0].reshape(4, -1))

        return fc_out