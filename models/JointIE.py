import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.modules import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from collections import defaultdict
from copy import deepcopy
import fewshot_re_kit

class JointIE(fewshot_re_kit.framework.IEModel):
    def __init__(self, 
                sentence_encoder, 
                hidden_size, 
                embedding_size, 
                context_layer, 
                context_dropout,
                span_repr_combination = 'x,y',
                max_span_width = 5):
        super().__init__(self, sentence_encoder)
        self.encoder = sentence_encoder
        self.contextual_layer = nn.LSTM(embedding_size, hidden_size, num_layers=context_layer, bidirectional=True, dropout=context_dropout)

        self._endpoint_span_extractor = EndpointSpanExtractor(hidden_size,
                                                                combination=span_repr_combination,
                                                                num_width_embeddings=max_span_width,
                                                                span_width_embedding_dim=span_width_embedding_dim,
                                                                bucket_widths=False)

        self._attentive_span_extractor = SelfAttentiveSpanExtractor(hidden_size)
                                                                    
    def forward(self, 
                tokens, # (batch_size, length)
                mask,   # (batch_size, length)
                converted_spans,  # (batch_size, span_num, 2)
                span_mask,        # (batch_size, span_num)
                ner_labels,       # (batch_size, span_num)
                relation_indices,  # (batch_size, span_pair_num, 2)
                relation_mask,     # (batch_size, span_pair_num)
                relation_labels):  # (batch_size, span_pair_num)
        



        return loss

