from joint_ie_kit.data_loader import get_loader
from joint_ie_kit.framework import IEFramework
from joint_ie_kit.sentence_encoder import BERTSentenceEncoder
from joint_ie_kit.utils import LabelField
import models
from models.JointIE import JointIE
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    ## File parameters
    parser.add_argument('--train', default='train',
            help='train file')
    parser.add_argument('--val', default='dev',
            help='val file')
    parser.add_argument('--test', default='test',
            help='test file')
    parser.add_argument('--root', default='./data',
            help='dataset root')
    parser.add_argument('--dataset', default='scierc',
            help='dataset')

    ## span
    parser.add_argument('--max_span_width', default=8, type=int,
            help='max number of word in a span')
    

    ## encoder
    parser.add_argument('--lexical_dropout', default=0.5, type=float,
            help='Embedding dropout')
    parser.add_argument('--embedding_size', default=768, type=float,
            help='Embedding dimension')
    
    ## model
    parser.add_argument('--model', default='JointIE',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: cnn or bert or roberta')
    parser.add_argument('--hidden_size', default=512, type=int,
           help='hidden size')
    parser.add_argument('--context_layer', default=1, type=int,
           help='number of contextual layers')
    parser.add_argument('--context_dropout', default=0, type=int,
           help='dropout rate in the contextual layer')
    parser.add_argument('--dropout', default=0.3, type=float,
           help='dropout rate')
    parser.add_argument('--span_width_dim', default=64, type=int,
           help='dimension of embedding for span width')
    parser.add_argument('--spans_per_word', default=0.6, type=float,
           help='thershold number of spans in each sentence')
    parser.add_argument('--e2e', default=True, action='store_false',
           help='Do not use End2Eng. End2End: if use gold relation index when training')

    ## Train
    parser.add_argument('--batch_size', default=16, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=20000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=1000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--lr', default=5e-5, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--optim', default='adamw',
           help='sgd / adam / adamw')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')
 
    opt = parser.parse_args()
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
        
    if encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        opt.embedding_size = 768
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout)
    else:
        raise NotImplementedError
    
    ner_label = LabelField()
    re_label = LabelField()
    
    root = os.path.join(opt.root, opt.dataset)
    train_data_loader = get_loader(root, opt.train, sentence_encoder, batch_size, ner_label, re_label, max_span_width=opt.max_span_width)
    val_data_loader = get_loader(root, opt.val, sentence_encoder, batch_size, ner_label, re_label, max_span_width=opt.max_span_width)
    test_data_loader = get_loader(root, opt.test, sentence_encoder, batch_size, ner_label, re_label, max_span_width=opt.max_span_width)

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    
    framework = IEFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val])
    
    if model_name == 'JointIE':
        model = JointIE(sentence_encoder, opt.hidden_size, opt.embedding_size, ner_label, 
                        re_label, opt.context_layer, opt.context_dropout, opt.dropout,
                        max_span_width=opt.max_span_width, span_width_embedding_dim=opt.span_width_dim,
                        spans_per_word=opt.spans_per_word, e2e=opt.e2e)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        framework.train(model, prefix, learning_rate=opt.lr,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                val_step=opt.val_step, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim)
    else:
        ckpt = opt.load_ckpt

    ner_f1, relation_f1 = framework.eval(model, opt.test_iter, ckpt=ckpt)
    print("RESULT: NER F1: {0:2.4f},  Relation F1: {1:2.4f}".format(ner_f1, relation_f1))

if __name__ == "__main__":
    main()
