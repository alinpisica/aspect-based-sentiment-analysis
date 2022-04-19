import torch

class BERT_Dropout_Linear(torch.nn.Module):
    def __init__(self, bert, dropout, no_out_labels, device='cpu'):
        super(BERT_Dropout_Linear, self).__init__()

        self.bert = bert.to(device)

        self.dropout = torch.nn.Dropout(dropout)

        self.out = torch.nn.Linear(self.bert.config.hidden_size, no_out_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _, pooled_output= self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)

        dropout_output = self.dropout(pooled_output)
        
        output = self.out(dropout_output)
        
        return output