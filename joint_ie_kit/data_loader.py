import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import warnings

class MissingDict(dict):
    """
    If key isn't there, returns default value. Like defaultdict, but it doesn't store the missing
    keys that were queried.
    """
    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val

def format_label_fields(ner, relations, sentence_start):
    
    ss = sentence_start
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start-ss, span_end-ss), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    # Relations
    relation_dict = MissingDict("",
        (
            ((  (span1_start-ss, span1_end-ss),  (span2_start-ss, span2_end-ss)   ), relation)
            for (span1_start, span1_end, span2_start, span2_end, relation) in relations
        )
    )

    return ner_dict, relation_dict


class DataLoader(data.Dataset):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, root, filename, encoder, batch_size, ner_label, re_label, max_span_width=5, context_width=1):
        self.batch_size = batch_size
        self.max_span_width = max_span_width
        
        self.ner_label = ner_label
        self.re_label = re_label

        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int( (context_width - 1) / 2)
        self.encoder = encoder
        self.lower = 'uncased' in encoder.encoder_type

        path = os.path.join(root, filename + ".json")

        data = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line))
        
        print(f'Begin processing {filename} dataset...')
        self.data = self.preprocess(data)
        print(f'Done. {filename} dataset has {len(self.data)} instances.')


    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for line in data:
            sentence_start = 0
            
            n_sentences = len(line["sentences"])
            # TODO(Ulme) Make it so that the
            line["sentence_groups"] = [[self._normalize_word(word) for sentence in line["sentences"][max(0, i-self.k):min(n_sentences, i + self.k + 1)] for word in sentence] for i in range(n_sentences)]
            line["sentence_start_index"] = [sum(len(line["sentences"][i-j-1]) for j in range(min(self.k, i))) if i > 0 else 0 for i in range(n_sentences)]
            line["sentence_end_index"] = [line["sentence_start_index"][i] + len(line["sentences"][i]) for i in range(n_sentences)]
            for sentence_group_nr in range(len(line["sentence_groups"])):
                if len(line["sentence_groups"][sentence_group_nr]) > 300:
                    line["sentence_groups"][sentence_group_nr] = line["sentences"][sentence_group_nr]
                    line["sentence_start_index"][sentence_group_nr] = 0
                    line["sentence_end_index"][sentence_group_nr] = len(line["sentences"][sentence_group_nr])
                    if len(line["sentence_groups"][sentence_group_nr])>300:
                        warnings.warn("Sentence with > 300 words; BERT may truncate.")
            
            zipped = zip(line["sentences"], line["ner"], line["relations"], line["sentence_groups"], line["sentence_start_index"], line["sentence_end_index"])

            for sentence_num, (sentence, ner, relations, groups, start_ix, end_ix) in enumerate(zipped):

                ner_dict, relation_dict = format_label_fields(ner, relations, sentence_start)
                sentence_start += len(sentence)
                sentence, spans, ner_labels, span_ner_labels, relation_indices, relation_labels = self.text_to_instance(sentence, ner_dict, relation_dict, sentence_num, groups, start_ix, end_ix, ner, relations)
                ##filter out sentences with only one entity.
                if len(span_ner_labels)<=1:
                    continue
                processed.append([sentence, spans, ner_labels, relation_indices, relation_labels])

        return processed

    def _normalize_word(self, word):

        if self.lower:
            word = word.lower()
        
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def text_to_instance(self,
                        sentence,
                        ner_dict,
                        relation_dict,
                        sentence_num,
                        groups,
                        start_ix,
                        end_ix,
                        ner,
                        relations
                        ):
        
        sentence = [self._normalize_word(word) for word in sentence]

        spans = []
        span_ner_labels = set()
        ner_labels = []
        for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
            span_ix = (start, end)
            spans.append((start, end))
            ner_label = ner_dict[span_ix]
            ner_labels.append(self.ner_label.get_id(ner_label))
            if ner_label:
                span_ner_labels.add(span_ix)
            
        n_spans = len(spans)
        candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i!=j]

        
        relation_indices = []
        relation_labels = []
        for i, j in candidate_indices:
            if spans[i] in span_ner_labels and spans[j] in span_ner_labels:
                span_pair = (spans[i], spans[j])
                relation_label = relation_dict[span_pair]
                
                relation_indices.append((i, j))
                relation_labels.append(self.re_label.get_id(relation_label))
        
        # Add negative re label
        self.re_label.get_id("")

        return sentence, spans, ner_labels, span_ner_labels, relation_indices, relation_labels


    def enumerate_spans(self, sentence, max_span_width, min_span_width=1):

        max_span_width = max_span_width or len(sentence)
        spans = []

        for start_index in range(len(sentence)):
            last_end_index = min(start_index + max_span_width, len(sentence))
            first_end_index = min(start_index + min_span_width - 1, len(sentence))
            for end_index in range(first_end_index, last_end_index):
                start = start_index
                end = end_index
                spans.append((start, end))
        return spans


    def __len__(self):
        return 100000000

    def __getitem__(self, index):
        
        index = random.randint(0, len(self.data)-1)
        sentence, spans, ner_labels, relation_indices, relation_labels = self.data[index]
        
        tokens, idx_dict = self.encoder.tokenize(sentence)
        
        converted_spans = []
        for span in spans:
            converted_spans.append(self.convert_span(span, idx_dict))
        
        
        return [tokens, converted_spans, ner_labels, relation_indices, relation_labels]

    
    def convert_span(self, span, idx_dict):

        start_idx = span[0]
        end_idx = span[1]
        
        span_idx = idx_dict[start_idx] + idx_dict[end_idx]

        return (min(span_idx), max(span_idx))

    def collate_fn(self, data):
        tokens_b, converted_spans_b, ner_labels_b, relation_indices_b, relation_labels_b = zip(*data)

        max_length = max([len(tokens) for tokens in tokens_b])
        ##padding
        for tokens in tokens_b:
            while len(tokens)<max_length:
                tokens.append(0)
        
        tokens_b = torch.LongTensor(tokens_b)
        ##mask
        mask = tokens_b.eq(0).eq(0).float()


        span_max_length = max([len(converted_spans) for converted_spans in converted_spans_b])
        ##span padding
        for converted_spans in converted_spans_b:
            while len(converted_spans)<span_max_length:
                converted_spans.append((0, 0))
        
        converted_spans_b = torch.LongTensor(converted_spans_b)
        ## span label padding
        for ner_labels in ner_labels_b:
            while len(ner_labels)<span_max_length:
                ner_labels.append(0)

        ner_labels_b = torch.LongTensor(ner_labels_b)
        ##span mask
        span_mask = converted_spans_b[:,:,0].eq(0).eq(0).float()

        

        ##relation padding
        relation_max_length = max([len(relation_indices) for relation_indices in relation_indices_b])
        for relation_indices in relation_indices_b:
            while len(relation_indices)<relation_max_length:
                relation_indices.append((0, 0))

        relation_indices_b = torch.LongTensor(relation_indices_b)

        ## relation label padding
        for relation_labels in relation_labels_b:
            while len(relation_labels)<relation_max_length:
                relation_labels.append(0)

        relation_labels_b = torch.LongTensor(relation_labels_b)

        ##relation mask (both indix are 0, then should be masked)
        relation_mask = torch.logical_and(relation_indices_b[:,:,0].eq(0), relation_indices_b[:,:,1].eq(0)).eq(0).float()

        return tokens_b, mask, converted_spans_b, span_mask, ner_labels_b, relation_indices_b, relation_mask, relation_labels_b

def get_loader(root, filename, encoder, batch_size, ner_label, re_label, max_span_width=5, context_width=1):

    dataset = DataLoader(root, filename, encoder, batch_size, ner_label, re_label, max_span_width=5, context_width=1)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            collate_fn=dataset.collate_fn)
    return iter(data_loader)
