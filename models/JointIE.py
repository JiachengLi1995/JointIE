import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.modules import EndpointSpanExtractor, SelfAttentiveSpanExtractor, FeedForward
from collections import defaultdict
from copy import deepcopy
import fewshot_re_kit

class JointIE(fewshot_re_kit.framework.IEModel):
    def __init__(self, 
                sentence_encoder, 
                hidden_size, 
                embedding_size, 
                ner_label,
                re_label,
                context_layer, 
                context_dropout=0.3,
                span_repr_combination = 'x,y',
                max_span_width = 5,
                span_width_embedding_dim = 64,
                spans_per_word = 0.6
                ):
        super().__init__(self, sentence_encoder)
        self.encoder = sentence_encoder
        self.contextual_layer = nn.LSTM(embedding_size, hidden_size, num_layers=context_layer, bidirectional=True, dropout=context_dropout)

        self.endpoint_span_extractor = EndpointSpanExtractor(hidden_size,
                                                                combination=span_repr_combination,
                                                                num_width_embeddings=max_span_width * 2,
                                                                span_width_embedding_dim=span_width_embedding_dim,
                                                                bucket_widths=False)

        self.attentive_span_extractor = SelfAttentiveSpanExtractor(hidden_size)

        ## span predictioin layer
        span_emb_dim = embedding_size+hidden_size
        ner_label_num = ner_label.get_num()
        self.span_layer = FeedForward(input_dim = span_emb_dim, num_layers=2, hidden_dim=hidden_size, dropout=context_dropout)
        self.span_proj_label = nn.Linear(hidden_size, ner_label_num)

        self.spans_per_word = spans_per_word
        self.ner_neg_id = ner_label.get_id('')

        self.re_neg_id = re_label.get_id('')

                                                                    
    def forward(self, 
                tokens, # (batch_size, length)
                mask,   # (batch_size, length)
                converted_spans,  # (batch_size, span_num, 2)
                span_mask,        # (batch_size, span_num)
                ner_labels,       # (batch_size, span_num)
                relation_indices,  # (batch_size, span_pair_num, 2)
                relation_mask,     # (batch_size, span_pair_num)
                relation_labels):  # (batch_size, span_pair_num)
        
        seq_len = tokens.size(1)


        embedding = self.sentence_encoder(tokens, mask) # (batch_size, length, bert_dim)
        contextual_embedding, _ = self.contextual_layer(embedding) #(batch_size, length, hidden_size)

        # extract span representation
        ep_span_emb = self.endpoint_span_extractor(contextual_embedding, converted_spans, span_mask)  #(batch_size , span_num, hidden_size)
        att_span_emb = self.attentive_span_extractor(embedding, converted_spans, span_mask)   #(batch_size, span_num, bert_dim)

        span_emb = torch.cat([ep_span_emb, att_span_emb] -1)  #(batch_size, span_num, hidden_size+bert_dim)

        span_logits = self.span_proj_label(self.span_layer(span_emb))  #(batch_size, span_num, span_label_num)
        span_prob = F.softmax(span_logits, dim=-1)  #(batch_size, span_num, span_label_num)

        span_prob_masked = self.prob_mask(span_prob, span_mask)  #(batch_size, span_num, span_label_num)

        num_spans_to_keep = self.get_num_spans_to_keep(self.spans_per_word, seq_len, span_prob.size(1))

        top_v = (-span_prob_masked[:, :, self.ner_neg_id]).topk(num_spans_to_keep, -1)[0][:, -1:]
        
        top_mask = span_prob[:, :, self.ner_neg_id] <= -top_v  #(batch_size, span_num)

        span_mask_subset = span_mask * (top_mask | ner_labels.ne(self.ner_neg_id)).float()

        return loss

    def prob_mask(self,
                  prob: torch.FloatTensor,
                  mask: torch.FloatTensor,
                  value: float = 1.0):
        ''' Add value to the positions masked out. prob is larger than mask by one dim. '''
        return prob + ((1.0 - mask) * value).unsqueeze(-1)

    def get_num_spans_to_keep(self,
                              spans_per_word,
                              seq_len,
                              max_value):
        
        if type(spans_per_word) is float:
            num_spans_to_keep = max(min(int(math.floor(spans_per_word * seq_len)), max_value), 1)
        elif type(spans_per_word) is int:
            num_spans_to_keep = max(min(spans_per_word, max_value), 1)
        else:
            raise ValueError
        return num_spans_to_keep

