import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class IEFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              ):
        
        print("Start training...")
    
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(model.parameters(),
                    learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()
     
        # Training
        best_ner_f1 = 0
        best_relation_f1 = 0
        patient = 0 # Stop training after several epochs without improvement.

        for it in range(start_iter, start_iter + train_iter):
            
            tokens_b, mask, converted_spans_b, span_mask, ner_labels_b, relation_indices_b, relation_mask, relation_labels_b = next(self.train_data_loader)
            
            output_dict  = model(tokens_b, mask, converted_spans_b, span_mask, ner_labels_b, relation_indices_b, relation_mask, relation_labels_b)

            loss = output_dict['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


            ner_results = output_dict['span_metrics']
            relation_results = output_dict['span_pair_metrics']
            
            ner_acc = ner_results[0].get_metric()
            ner_prf = ner_results[1].get_metric()
            ner_prf_b = ner_results[2].get_metric()

            relation_acc = relation_results[0].get_metric()
            relation_prf = relation_results[1].get_metric()
            relation_prf_b = relation_results[2].get_metric()
         
            
            sys.stdout.write('step: {0:4} | loss: {1:2.6f},  NER_acc: {2:3.2f},  RE_acc: {3:3.2f}%'.format(it + 1, loss, 100 * ner_acc, 100 * relation_acc) +'\r')
            sys.stdout.write('NER \t F1 \t Precision \t Recall %')
            sys.stdout.write('prf \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} %'.format(ner_prf['f'], ner_prf['p'], ner_prf['r']) +'\r')
            sys.stdout.write('prf_b \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} %'.format(ner_prf_b['f'], ner_prf_b['p'], ner_prf_b['r']) +'\r')
            sys.stdout.write('Relation \t F1 \t Precision \t Recall %')
            sys.stdout.write('prf \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} %'.format(relation_prf['f'], relation_prf['p'], relation_prf['r']) +'\r')
            sys.stdout.write('prf_b \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} %'.format(relation_prf_b['f'], relation_prf_b['p'], relation_prf_b['r']) +'\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                model.metric_reset()
                ner_f1, relation_f1 = self.eval(model, val_iter)
                model.train()
                if ner_f1 > best_ner_f1 or relation_f1 > best_relation_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_ner_f1 = ner_f1
                    best_relation_f1 = relation_f1
                model.metric_reset()
        model.metric_reset()
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            eval_iter,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        with torch.no_grad():
            for it in range(eval_iter):
                
                tokens_b, mask, converted_spans_b, span_mask, ner_labels_b, relation_indices_b, relation_mask, relation_labels_b = next(self.train_data_loader)
            
                output_dict  = model(tokens_b, mask, converted_spans_b, span_mask, 
                                    ner_labels_b, relation_indices_b, relation_mask, relation_labels_b)

                loss = output_dict['loss']

                ner_results = output_dict['span_metrics']
                relation_results = output_dict['span_pair_metrics']
                
                ner_acc = ner_results[0].get_metric()
                ner_prf = ner_results[1].get_metric()
                ner_prf_b = ner_results[2].get_metric()

                relation_acc = relation_results[0].get_metric()
                relation_prf = relation_results[1].get_metric()
                relation_prf_b = relation_results[2].get_metric()

                sys.stdout.write('step: {0:4} | loss: {1:2.6f},  NER_acc: {2:3.2f},  RE_acc: {3:3.2f}%'.format(it + 1, loss, 100 * ner_acc, 100 * relation_acc) +'\r')
                sys.stdout.write('NER \t F1 \t Precision \t Recall')
                sys.stdout.write('prf \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(ner_prf['f'], ner_prf['p'], ner_prf['r']) +'\r')
                sys.stdout.write('prf_b \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(ner_prf_b['f'], ner_prf_b['p'], ner_prf_b['r']) +'\r')
                sys.stdout.write('Relation \t F1 \t Precision \t Recall')
                sys.stdout.write('prf \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(relation_prf['f'], relation_prf['p'], relation_prf['r']) +'\r')
                sys.stdout.write('prf_b \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(relation_prf_b['f'], relation_prf_b['p'], relation_prf_b['r']) +'\r')
                sys.stdout.flush()
                
            print("")
        return ner_prf['f'], relation_prf['f']