import math
import copy
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.t_encoder = TextEncoder(*param.encoder_param.text)
        self.v_encoder = Encoder(*param.encoder_param.visual)
        self.a_encoder = Encoder(*param.encoder_param.audio)
        self.v_interact = InteractLayer(*param.interact_param.visual)
        self.a_interact = InteractLayer(*param.interact_param.audio)
        self.num_interactions = param.num_interactions
        t_dim = param.encoder_param.text[1]
        v_dim = param.interact_param.visual[0]
        a_dim = param.interact_param.audio[0]
        cat_dim = t_dim + v_dim + a_dim
        self.t_regression = nn.Linear(t_dim, 1)
        self.v_regression = nn.Linear(v_dim, 1)
        self.a_regression = nn.Linear(a_dim, 1)
        self.cat_regression = nn.Linear(cat_dim, 1)
        self.w_cat = nn.Parameter(torch.tensor(1.0))
        self.w_t = nn.Parameter(torch.tensor(1.0))
        self.w_v = nn.Parameter(torch.tensor(0.01))
        self.w_a = nn.Parameter(torch.tensor(0.05))

    def forward(self, input_ids, t_masks, visual, va_masks, audio):
        t_encoded = self.t_encoder(input_ids, t_masks)
        v_encoded = self.v_encoder(visual, va_masks)
        a_encoded = self.a_encoder(audio, va_masks)
        for i in range(self.num_interactions):
            v_output = self.v_interact(v_encoded, t_encoded, va_masks, t_masks)
            a_output = self.a_interact(a_encoded, t_encoded, va_masks, t_masks)
            v_encoded, a_encoded = v_output, a_output
        t_utter = torch.mean(t_encoded, 0)
        v_utter = torch.mean(v_encoded, 0)
        a_utter = torch.mean(a_encoded, 0)
        cat_utter = torch.cat([t_utter, v_utter, a_utter], -1)
        t_res = self.t_regression(t_utter).squeeze(1)
        v_res = self.v_regression(v_utter).squeeze(1)
        a_res = self.a_regression(a_utter).squeeze(1)
        cat_res = self.cat_regression(cat_utter).squeeze(1)
        return cat_res, t_res, v_res, a_res


class TextEncoder(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(TextEncoder, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.linear = nn.Linear(d_model, dim_feedforward)

    def forward(self, input_ids, t_masks):
        outputs = self.model(input_ids, t_masks)
        encoded = outputs[0]
        encoded = self.linear(encoded)
        return encoded.transpose(0, 1)


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers):
        super().__init__()
        self.num_heads = nhead
        self.pe = PositionalEncoding(d_model, dropout)
        self.layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(self.layer, num_layers)
        self.linear = nn.Linear(d_model, dim_feedforward)

    def forward(self, inputs, attn_mask):
        inputs = self.pe(inputs)
        attn_mask = compute_mask(attn_mask, attn_mask, self.num_heads)
        encoded = self.encoder(inputs, mask=attn_mask)
        encoded = self.linear(encoded)
        return encoded


class InteractLayer(nn.Module):
    def __init__(self, d_model, dim_1, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.num_heads = nhead
        self.multihead_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=dim_1, vdim=dim_1)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)

    def forward(self, encoded, memory1, mask_q, mask_k1):
        attn_mask_1 = compute_mask(mask_q, mask_k1, self.num_heads)
        inter1 = self.multihead_attn_1(encoded, memory1, memory1, attn_mask=attn_mask_1)[0]
        attn1 = self.add_norm_1(encoded, inter1)
        ff = self.ff(attn1)
        output = self.add_norm_2(attn1, ff)
        return output


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prior, after):
        return self.norm(prior + self.dropout(after))


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, inputs):
        return self.linear2(self.dropout(self.activation(self.linear1(inputs))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def drop_path(paths, drop_rates):
    lens = len(paths)
    drop_rates = torch.tensor(drop_rates)
    drop = torch.bernoulli(drop_rates)
    if torch.all(drop == 0):
        idx = randint(0, lens-1)
        output = paths[idx]
    else:
        output = sum([paths[i] * drop[i] for i in range(lens)]) / torch.sum(drop)
    return output


def compute_mask(mask_1, mask_2, num_heads):
    mask_1 = torch.unsqueeze(mask_1, 2)
    mask_2 = torch.unsqueeze(mask_2, 1)
    attn_mask = torch.bmm(mask_1, mask_2)
    attn_mask = attn_mask.repeat(num_heads, 1, 1)
    return attn_mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
