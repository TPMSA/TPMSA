import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.t_encoder = TextEncoder(*param.encoder_param.text)
        concat_dim = param.encoder_param.text[1]
        self.regression = nn.Linear(concat_dim, 1)

    def forward(self, input_ids, t_masks):
        t_encoded = self.t_encoder(input_ids, t_masks)
        t_utter = torch.mean(t_encoded, 0)
        fusion = t_utter
        results = self.regression(fusion).squeeze(1)
        return results


class TextEncoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super(TextEncoder, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, input_ids, t_masks):
        outputs = self.model(input_ids, t_masks)
        encoded = outputs[0]
        encoded = self.linear(encoded)
        return encoded.transpose(0, 1)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
