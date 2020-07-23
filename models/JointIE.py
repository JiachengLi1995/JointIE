import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.modules import EndpointSpanExtractor, SelfAttentiveSpanExtractor, FeedForward, \
                        flatten_and_batch_shift_indices, batched_index_select, SpanPairPairedLayer
from joint_ie_kit.categorical_accuracy import CategoricalAccuracy
from joint_ie_kit.precision_recall_f1 import PrecisionRecallF1
from collections import defaultdict
from copy import deepcopy
import joint_ie_kit

class JointIE(nn.Module):
    def __init__(self, 
                sentence_encoder, 
                hidden_size, 
                embedding_size, 
                ner_label,
                re_label,
                context_layer, 
                context_dropout=0.3,
                dropout = 0.3,
                span_repr_combination = 'x,y',
                max_span_width = 5,
                span_width_embedding_dim = 64,
                spans_per_word = 0.6,
                e2e = True
                ):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.contextual_layer = nn.LSTM(embedding_size, embedding_size//2, num_layers=context_layer, bidirectional=True, dropout=context_dropout)

        self.endpoint_span_extractor = EndpointSpanExtractor(embedding_size,
                                                                combination=span_repr_combination,
                                                                num_width_embeddings=max_span_width * 5,
                                                                span_width_embedding_dim=span_width_embedding_dim,
                                                                bucket_widths=False)

        self.attentive_span_extractor = SelfAttentiveSpanExtractor(embedding_size)

        ## span predictioin layer
        span_emb_dim = self.endpoint_span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim()
        ner_label_num = ner_label.get_num()
        self.span_layer = FeedForward(input_dim = span_emb_dim, num_layers=2, hidden_dim=hidden_size, dropout=dropout)
        self.span_proj_label = nn.Linear(hidden_size, ner_label_num)

        self.spans_per_word = spans_per_word
        self.ner_neg_id = ner_label.get_id('')

        self.re_neg_id = re_label.get_id('')
        self.e2e = e2e

        ## span pair
        re_label_num = re_label.get_num()
        dim_reduce_layer = FeedForward(span_emb_dim, num_layers = 1, hidden_dim = hidden_size)
        repr_layer = FeedForward(hidden_size * 3 + span_width_embedding_dim, num_layers = 2, hidden_dim = hidden_size//4)
        self.span_pair_layer = SpanPairPairedLayer(dim_reduce_layer, repr_layer)
        self.span_pair_label_proj = nn.Linear(hidden_size//4, re_label_num)

        ## metrics
        # ner
        self.ner_acc = CategoricalAccuracy(top_k=1, tie_break=False)
        self.ner_prf = PrecisionRecallF1(neg_label=self.ner_neg_id)
        self.ner_prf_b = PrecisionRecallF1(neg_label=self.ner_neg_id, binary_match=True)

        # relation

        self.re_acc = CategoricalAccuracy(top_k=1, tie_break=False)
        self.re_prf = PrecisionRecallF1(neg_label=self.re_neg_id)
        self.re_prf_b = PrecisionRecallF1(neg_label=self.re_neg_id, binary_match=True)


                                                                    
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
        spans_num = converted_spans.size(1)

        embedding = self.sentence_encoder(tokens, mask) # (batch_size, length, bert_dim)
        contextual_embedding, _ = self.contextual_layer(embedding) #(batch_size, length, hidden_size)

        # extract span representation
        ep_span_emb = self.endpoint_span_extractor(contextual_embedding, converted_spans, span_mask)  #(batch_size , span_num, hidden_size)
        att_span_emb = self.attentive_span_extractor(embedding, converted_spans, span_mask)   #(batch_size, span_num, bert_dim)
    
        span_emb = torch.cat((ep_span_emb, att_span_emb), dim = -1)  #(batch_size, span_num, hidden_size+bert_dim)

        span_logits = self.span_proj_label(self.span_layer(span_emb))  #(batch_size, span_num, span_label_num)
        span_prob = F.softmax(span_logits, dim=-1)  #(batch_size, span_num, span_label_num)
        _, span_pred = span_prob.max(2)

        span_prob_masked = self.prob_mask(span_prob, span_mask)  #(batch_size, span_num, span_label_num)

        if self.training:
            num_spans_to_keep = self.get_num_spans_to_keep(self.spans_per_word, seq_len, span_prob.size(1))
            top_v = (-span_prob_masked[:, :, self.ner_neg_id]).topk(num_spans_to_keep, -1)[0][:, -1:]
            top_mask = span_prob[:, :, self.ner_neg_id] <= -top_v  #(batch_size, span_num)
            span_mask_subset = span_mask * (top_mask | ner_labels.ne(self.ner_neg_id)).float()

        else:
                
            span_mask_subset = span_mask

        span_neg_logit = span_logits[:, :, self.ner_neg_id]

        span_loss = sequence_cross_entropy_with_logits(
                    span_logits, ner_labels, span_mask_subset,
                    average='sum')

        span_len = converted_spans[:, :, 1] - converted_spans[:, :, 0] + 1

        ## span metrics
        self.ner_acc(span_logits, ner_labels, span_mask_subset)
        self.ner_prf(span_logits.max(-1)[1], ner_labels, span_mask_subset.long(), bucket_value=span_len)
        self.ner_prf_b(span_logits.max(-1)[1], ner_labels, span_mask_subset.long())

        
        # span pair (relation)

        if not self.e2e:
            # use provided span pairs to construct embedding
            span_pair_mask = relation_mask
            # get span pair embedding
            # SHAPE: (batch_size * num_span_pairs * 2)
            flat_span_pairs = flatten_and_batch_shift_indices(relation_indices, spans_num)

            # span pair prediction
            # SHAPE: (batch_size, num_span_pairs, num_classes)
            span_pair_logits = self.span_pair_label_proj(self.span_pair_layer(span_emb, relation_indices))

            # get negative span logits of the pair by sum
            # SHAPE: (batch_size, num_span_pairs, 2)
            span_pair_neg_logit = batched_index_select(
                                span_neg_logit.unsqueeze(-1), relation_indices, flat_span_pairs).squeeze(-1)
            # SHAPE: (batch_size, num_span_pairs)
            span_pair_neg_logit = span_pair_neg_logit.sum(-1)
            # SHAPE: (batch_size, num_span_pairs, 2)
            span_pair_len = batched_index_select(span_len.unsqueeze(-1), relation_indices, flat_span_pairs).squeeze(-1)
            # SHAPE: (batch_size, num_span_pairs)
            span_pair_len = span_pair_len.max(-1)[0]

        else:
            # select span pairs by span scores to construct embedding

            ref_span_pairs = relation_indices
            ref_span_pair_mask = relation_mask

            # rank spans
            # TODO: sequences in the same batch keeps the same number of spans despite their different length
            num_spans_to_keep = self.get_num_spans_to_keep(self.spans_per_word, seq_len, span_prob.size(1))
            
                # SHAPE: (task_batch_size, num_spans_to_keep)
            _, top_ind = (-span_prob_masked[:, :, self.ner_neg_id]).topk(num_spans_to_keep, -1)
            # sort according to the order (not strictly based on order because spans overlap)
            top_ind = top_ind.sort(-1)[0]

            # get out-of-bound mask
            # TODO: span must be located at the beginning
            # SHAPE: (task_batch_size, num_spans_to_keep)
            top_ind_mask = top_ind < span_mask.sum(-1, keepdim=True).long()

            # get pairs
            num_spans_to_keep = top_ind.size(1)
            external2internal = self.extenral_to_internal(top_ind, spans_num)

            # SHAPE: (batch_size * num_span_pairs * 2)
            span_pairs, span_pair_mask, span_pair_shape = self.span_ind_to_pair_ind(
                top_ind, top_ind_mask, method=None, absolute=False)

            span_pairs_internal = external2internal(span_pairs)

            # get negative span logits of the pair by sum
            # SHAPE: (batch_size * num_span_pairs * 2)
            flat_span_pairs = flatten_and_batch_shift_indices(span_pairs, spans_num)
            # SHAPE: (batch_size, num_span_pairs, 2)
            span_pair_neg_logit = batched_index_select(
                span_neg_logit.unsqueeze(-1), span_pairs, flat_span_pairs).squeeze(-1)
            # SHAPE: (batch_size, num_span_pairs)
            span_pair_neg_logit = span_pair_neg_logit.sum(-1)
            # SHAPE: (batch_size, num_span_pairs, 2)
            span_pair_len = batched_index_select(
                span_len.unsqueeze(-1), span_pairs, flat_span_pairs).squeeze(-1)
            # SHAPE: (batch_size, num_span_pairs)
            span_pair_len = span_pair_len.max(-1)[0]

            # get span kept
            # SHAPE: (batch_size * num_spans_to_keep)
            flat_top_ind = flatten_and_batch_shift_indices(top_ind, spans_num)
            # SHAPE: (batch_size, num_spans_to_keep, 2)
            spans_for_pair = batched_index_select(converted_spans, top_ind, flat_top_ind)
            # SHAPE: (batch_size, num_spans_to_keep, span_emb_dim)
            span_emb_for_pair = batched_index_select(span_emb, top_ind, flat_top_ind)
            # SHAPE: (batch_size, num_spans_to_keep, num_classe)
            span_prob_for_pair = batched_index_select(span_prob, top_ind, flat_top_ind)

            # span pair prediction
            # SHAPE: (batch_size, num_span_pairs, num_classes)
            span_pair_logits = self.span_pair_label_proj(self.span_pair_layer(span_emb_for_pair, span_pairs_internal))

        span_pair_prob = F.softmax(span_pair_logits, dim=-1)
        _, span_pair_pred = span_pair_prob.max(2)

        span_pair_mask_for_loss = span_pair_mask

        if not self.e2e:
            # SHAPE: (task_batch_size, num_span_pairs)
            span_pair_labels = relation_labels
        else:
            # SHAPE: (task_batch_size, num_span_pairs)
            ref_span_pair_labels = relation_labels
            # SHAPE: (task_batch_size, num_spans_to_keep * num_spans_to_keep)
            span_pair_labels = self.label_span_pair(span_pairs, ref_span_pairs, ref_span_pair_labels, ref_span_pair_mask)

        span_pair_loss = sequence_cross_entropy_with_logits(
                        span_pair_logits, span_pair_labels, span_pair_mask_for_loss,
                        average='sum')

        ## relation metrics

        self.re_acc(span_pair_logits, span_pair_labels, span_pair_mask)
        if not self.e2e:
            recall = None
        else:
            recall = ref_span_pair_labels.ne(self.re_neg_id)
            recall = (recall.float() * ref_span_pair_mask).long()    

        self.re_prf(span_pair_pred, span_pair_labels, span_pair_mask.long(), recall=recall, bucket_value=span_pair_len)
        self.re_prf_b(span_pair_pred, span_pair_labels, span_pair_mask.long(), recall=recall)

        ## loss
        loss = span_loss + span_pair_loss

        ## output dict

        output_dict = {
            'loss': loss,
            'span_loss': span_loss,
            'span_pred': span_pred,
            'span_metrics': [self.ner_acc, self.ner_prf, self.ner_prf_b],
            'span_pair_loss': span_pair_loss,
            'span_pair_pred': span_pair_pred,
            'span_pair_metrics':[self.re_acc, self.re_prf, self.re_prf_b]
        }

        return output_dict

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

    def extenral_to_internal(self,
                             span_ind: torch.LongTensor,  # SHAPE: (batch_size, num_spans)
                             total_num_spans: int,
                             ):  # SHAPE: (batch_size, total_num_spans)
        batch_size, num_spans = span_ind.size()
        # SHAPE: (batch_size, total_num_spans)
        converter = span_ind.new_zeros((batch_size, total_num_spans))
        new_ind = torch.arange(num_spans, device=span_ind.device).unsqueeze(0).repeat(batch_size, 1)
        # SHAPE: (batch_size, total_num_spans)
        converter.scatter_(-1, span_ind, new_ind)
        def converter_(ind):
            flat_ind = flatten_and_batch_shift_indices(ind, total_num_spans)
            new_ind = batched_index_select(converter.unsqueeze(-1), ind, flat_ind).squeeze(-1)
            return new_ind  # the same shape as ind
        return converter_

    def span_ind_to_pair_ind(self,
                             span_ind: torch.LongTensor,  # SHAPE: (batch_size, num_spans)
                             span_ind_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_spans)
                             start_span_ind: torch.LongTensor = None,  # SHAPE: (batch_size, num_spans2)
                             start_span_ind_mask: torch.FloatTensor = None,  # SHAPE: (batch_size, num_spans2)
                             method: str = None,
                             absolute: bool = True):
        ''' Create span pair indices and corresponding mask based on selected spans '''
        batch_size, num_spans = span_ind.size()

        if method and method.startswith('left:'):
            left_size = int(method.split(':', 1)[1])

            # get mask
            # span indices should be in the same order as they appear in the sentence
            if absolute:
                # SHAPE: (batch_size, num_spans, num_spans)
                left_mask = (span_ind.unsqueeze(1) < span_ind.unsqueeze(2)) & \
                            (span_ind.unsqueeze(1) >= (span_ind.unsqueeze(2) - left_size))
            else:
                # SHAPE: (num_spans,)
                end_boundary = torch.arange(num_spans, device=span_ind.device)
                start_boundary = end_boundary - left_size
                # SHAPE: (num_spans, num_spans)
                left_mask = (end_boundary.unsqueeze(0) < end_boundary.unsqueeze(-1)) & \
                            (end_boundary.unsqueeze(0) >= start_boundary.unsqueeze(-1))
                left_mask = left_mask.unsqueeze(0).repeat(batch_size, 1, 1)

            # SHAPE: (batch_size, num_spans)
            left_mask_num = left_mask.sum(-1)
            left_mask_num_max = max(left_mask_num.max().item(), 1)  # keep at least 1 span pairs to avoid bugs
            # SHAPE: (batch_size, num_spans)
            left_mask_num_left = left_mask_num_max - left_mask_num
            # SHAPE: (1, 1, left_mask_num_max)
            left_mask_ext = torch.arange(left_mask_num_max, device=span_ind.device).unsqueeze(0).unsqueeze(0)
            # SHAPE: (batch_size, num_spans, left_mask_num_max)
            left_mask_ext = left_mask_ext < left_mask_num_left.unsqueeze(-1)
            # SHAPE: (batch_size, num_spans, num_spans + left_mask_num_max)
            left_mask = torch.cat([left_mask, left_mask_ext], -1)

            # extend span_ind and span_ind_mask
            # SHAPE: (batch_size, num_spans + left_mask_num_max)
            span_ind_child = torch.cat([span_ind,
                                        span_ind.new_zeros((batch_size, left_mask_num_max))], -1)
            span_ind_child_mask = torch.cat([span_ind_mask,
                                             span_ind_mask.new_zeros((batch_size, left_mask_num_max))], -1)
            # SHAPE: (batch_size, num_spans, left_mask_num_max)
            span_ind_child = span_ind_child.unsqueeze(1).masked_select(left_mask).view(
                batch_size, num_spans, left_mask_num_max)
            span_ind_child_mask = span_ind_child_mask.unsqueeze(1).masked_select(left_mask).view(
                batch_size, num_spans, left_mask_num_max)

            # concat with parent ind
            span_pairs = torch.stack([span_ind.unsqueeze(2).repeat(1, 1, left_mask_num_max),
                                      span_ind_child], -1)
            span_pair_mask = torch.stack([span_ind_mask.unsqueeze(2).repeat(1, 1, left_mask_num_max),
                                          span_ind_child_mask], -1) > 0
            # SHAPE: (batch_size, num_spans * left_mask_num_max, 2)
            span_pairs = span_pairs.view(-1, num_spans * left_mask_num_max, 2)
            # SHAPE: (batch_size, num_spans * left_mask_num_max)
            span_pair_mask = span_pair_mask.view(-1, num_spans * left_mask_num_max, 2).all(-1).float()

            # TODO: Because of span_ind_mask, the result might not have left_size spans.
            #   This problem does not exist when the spans are all located at the top of the tensor
            return span_pairs, span_pair_mask, (num_spans, left_mask_num_max)

        if method == 'gold_predicate':
            _, num_spans2 = start_span_ind.size()
            # default: compose num_spans2 * num_spans pairs
            span_pairs = torch.stack([start_span_ind.unsqueeze(2).repeat(1, 1, num_spans),
                                      span_ind.unsqueeze(1).repeat(1, num_spans2, 1)], -1)
            span_pair_mask = torch.stack([start_span_ind_mask.unsqueeze(2).repeat(1, 1, num_spans),
                                          span_ind_mask.unsqueeze(1).repeat(1, num_spans2, 1)], -1)
            # SHAPE: (batch_size, num_spans2 * num_spans, 2)
            span_pairs = span_pairs.view(-1, num_spans2 * num_spans, 2)
            # SHAPE: (batch_size, num_spans * num_spans)
            span_pair_mask = span_pair_mask.view(-1, num_spans2 * num_spans, 2).all(-1).float()
            return span_pairs, span_pair_mask, (num_spans2, num_spans)

        # default: compose num_spans * num_spans pairs
        span_pairs = torch.stack([span_ind.unsqueeze(2).repeat(1, 1, num_spans),
                                  span_ind.unsqueeze(1).repeat(1, num_spans, 1)], -1)
        span_pair_mask = torch.stack([span_ind_mask.unsqueeze(2).repeat(1, 1, num_spans),
                                      span_ind_mask.unsqueeze(1).repeat(1, num_spans, 1)], -1)
        # SHAPE: (batch_size, num_spans * num_spans, 2)
        span_pairs = span_pairs.view(-1, num_spans * num_spans, 2)
        # SHAPE: (batch_size, num_spans * num_spans)
        span_pair_mask = span_pair_mask.view(-1, num_spans * num_spans, 2).all(-1).float()
        return span_pairs, span_pair_mask, (num_spans, num_spans)


    def label_span_pair(self,
                        span_pairs: torch.IntTensor,  # SHAPE: (batch_size, num_span_pairs1, 2)
                        ref_span_pairs: torch.IntTensor,  # SHAPE: (batch_size, num_span_pairs2, 2)
                        ref_span_pair_labels: torch.IntTensor,  # SHPAE: (batch_size, num_span_pairs2)
                        ref_span_pair_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_span_pairs2)
                        spans: torch.IntTensor = None,  # SHAPE: (batch_size, num_spans, 2)
                        use_binary: bool = False,
                        span_pair_pred: torch.IntTensor = None  # SHAPE: (batch_size, num_span_pairs1)
                        ): # SHPAE: (batch_size, num_span_pairs1)
        neg_label_ind = self.re_neg_id
        device = span_pairs.device
        span_pairs = span_pairs.cpu().numpy()
        ref_span_pairs = ref_span_pairs.cpu().numpy()
        ref_span_pair_labels = ref_span_pair_labels.cpu().numpy()
        ref_span_pair_mask = ref_span_pair_mask.cpu().numpy()
        batch_size = ref_span_pairs.shape[0]
        ref_num_span_pairs = ref_span_pairs.shape[1]
        num_span_pairs = span_pairs.shape[1]

        if spans is not None and use_binary:
            spans = spans.cpu().numpy()
            if span_pair_pred is not None:
                span_pair_pred = span_pair_pred.cpu().numpy()

        span_pair_labels = []
        for b in range(batch_size):
            label_dict = defaultdict(lambda: neg_label_ind)
            label_dict.update(dict((tuple(ref_span_pairs[b, i]), ref_span_pair_labels[b, i])
                                   for i in range(ref_num_span_pairs) if ref_span_pair_mask[b, i] > 0))
            labels = []
            for i in range(num_span_pairs):
                tsp1, tsp2 = tuple(span_pairs[b, i])
                assign_label = label_dict[(tsp1, tsp2)]
                if span_pair_pred is not None:
                    pred_label = span_pair_pred[b, i]
                else:
                    pred_label = None
                if pred_label == neg_label_ind:  # skip pairs not predicated as positive
                    labels.append(assign_label)
                    continue
                if spans is not None and use_binary:
                    # find overlapping span pairs
                    has_overlap = False
                    for (sp1, sp2), l in label_dict.items():
                        if l == neg_label_ind:
                            continue
                        if pred_label and l != pred_label:  # only look at ground truth with predicted label
                            continue
                        if self.has_overlap(spans[b, tsp1], spans[b, sp1]) and \
                                self.has_overlap(spans[b, tsp2], spans[b, sp2]):
                            assign_label = l
                            has_overlap = True
                labels.append(assign_label)
            span_pair_labels.append(labels)
        return torch.LongTensor(span_pair_labels).to(device)
    
    def metric_reset(self):

        self.ner_acc.reset()
        self.ner_prf.reset()
        self.ner_prf_b.reset()

        self.re_acc.reset()
        self.re_prf.reset()
        self.re_prf_b.reset()


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.FloatTensor:
    if average not in {None, "token", "batch", "sum", "batch_sum"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', 'batch', 'sum', or 'batch_sum'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    elif average == "sum":
        return negative_log_likelihood.sum()
    elif average == "batch_sum":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss.sum()
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss

