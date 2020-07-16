import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel

class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, lexical_dropout): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.encoder_type = pretrain_path

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        
    def forward(self, tokens, mask):
        
        outputs, _ = self.bert(tokens, attention_mask=mask)
        outputs = self._lexical_dropout(outputs)

        return outputs
    
    def tokenize(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        idx_dict = dict()
        cur_pos = 1

        for i, token in enumerate(raw_tokens):
            
            sub_words = self.tokenizer.tokenize(token)
            tokens += sub_words
            idx_dict[i] = list(range(cur_pos, cur_pos+len(sub_words)))
            cur_pos += len(sub_words)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens, idx_dict


