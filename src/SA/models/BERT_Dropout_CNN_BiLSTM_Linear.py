import torch

class BERT_Dropout_CNN_BiLSTM_Linear(torch.nn.Module):
    def __init__(self, bert, bert_seq_len, dropout, bilstm_in_features, no_out_labels, conv_out_channels, conv_kernel_size, device='cpu', batch_size=4, conv_dilation=1, conv_padding=0, conv_stride=1):
        super(BERT_Dropout_CNN_BiLSTM_Linear, self).__init__()          
        
        self.device = device
        self.batch_size = batch_size

        self.bert = bert

        self.dropout = torch.nn.Dropout(dropout)

        self.conv = torch.nn.Conv1d(self.bert.config.hidden_size, conv_out_channels, conv_kernel_size, dilation=conv_dilation, padding=conv_padding)

        conv_l_out_size = (bert_seq_len + 2 * conv_padding - conv_dilation * (conv_kernel_size - 1) - 1) // conv_stride + 1

        self.lstm = torch.nn.LSTM(conv_l_out_size, bilstm_in_features // 2, bidirectional=True, batch_first=True)

        self.fc = torch.nn.Linear(bilstm_in_features * conv_out_channels, no_out_labels)
    
    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        dropout_output = self.dropout(last_hidden_state.to(self.device)).to(self.device).permute(0, 2, 1)
        
        conv_output = self.conv(dropout_output).to(self.device)

        lstm_output = self.lstm(conv_output)
        
        output = self.fc(lstm_output[0].reshape(self.batch_size, -1))

        return output