import torch
import torch.nn as nn

class BERTClass(torch.nn.Module):
    def __init__(self, embedding_model, embedding_dim, no_out_labels, droput=0.3, device='cpu'):
        super(BERTClass, self).__init__()

        self.embedding = embedding_model.to(device)

        self.dropout = torch.nn.Dropout(droput)

        self.out = torch.nn.Linear(embedding_dim, no_out_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.embedding(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)

        output_2 = self.dropout(output_1)
        
        output = self.out(output_2)
        
        return output