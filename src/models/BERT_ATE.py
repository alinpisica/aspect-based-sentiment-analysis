import torch
from transformers import BertModel

class BERT_ATE(torch.nn.Module):
    def __init__(self, pretrained_bert_model_variant):
        super(BERT_ATE, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_bert_model_variant)

        # 3 is the number of possible IOB outputs:
        # 0 for OUTSIDE
        # 1 for BEGINNING
        # 2 for INSIDE
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, ids_tensors, masks_tensors):
        bert_outputs,_ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)

        linear_outputs = self.linear(bert_outputs)

        return linear_outputs
