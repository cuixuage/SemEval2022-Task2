import re
import csv
import os
import json
import random
from collections import defaultdict
from typing import Any, Optional, Dict, Iterable, List, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F

from collections   import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenise_idiom( phrase ) :
    
    return 'ID' + re.sub( r'[\s|-]', '', phrase ).lower() + 'ID'

if __name__ == '__main__' :
    print( tokenise_idiom( 'big fish' ) )
    print( tokenise_idiom( 'alta-costura' ) )
    print( tokenise_idiom( 'pastoralemã' ) )
    assert tokenise_idiom( 'big fish' ) == 'IDbigfishID'
    assert tokenise_idiom( 'alta-costura' ) == 'IDaltacosturaID'
    assert tokenise_idiom( 'pão-duro' ) == 'IDpãoduroID'
    assert tokenise_idiom( 'pastoralemão' ) == 'IDpastoralemãoID'
    print( "All good" )
    

def match_idioms( idiom_word_dict, sentence ) :

    sentence_words = word_tokenize( sentence )
    new_sentence_words = list()
    for word in sentence_words :
        if not re.match( r'^\'\w', word ) is None :
            new_sentence_words.append( "'" )
            word = re.sub( r'^\'', '', word )
            new_sentence_words.append( word )
            continue
        new_sentence_words.append( word )
    sentence_words = new_sentence_words

    matched_idioms = list()
    for index in range( len( sentence_words ) - 1 ) :
        this_word   = sentence_words[ index ].lower() 
        idiom_words = idiom_word_dict[ this_word ]
        if len( idiom_words ) == 0 :
            continue
        next_word = sentence_words[ index + 1 ].lower()
        for idiom_word_2 in idiom_words :
            if idiom_word_2.lower() == next_word or idiom_word_2 + 's' == next_word or idiom_word_2 == '*' :
                matched_idioms.append( this_word + ' ' + idiom_word_2  )
                
    return matched_idioms

def create_idiom_word_dict( idioms ) : 
    
    idiom_word_dict = defaultdict( list )
    for idiom in idioms :
        split_idiom = idiom.split()
        if len( split_idiom ) == 2 : 
            word1, word2 = split_idiom
        elif len( split_idiom ) == 1 : 
            word1 = split_idiom[0]
            word2 = '*'
        else : 
            raise Exception( "Cannot handle length!" )

        idiom_word_dict[ word1 ].append( word2 ) 
        
    return idiom_word_dict


def _load_csv( location, delimiter, is_gz=False ) :

    if delimiter is None :
        with open( location ) as infile :
            data = infile.read().lstrip().rstrip().split( '\n' )
            header = data[0]
            data   = data[1:]
        return header, data
            
    
    header = None
    data   = list()

    csvfile = reader = None
    if is_gz : 
        csvfile = gzip.open( location, 'rt', encoding='utf8' ) 
        reader = csv.reader( csvfile, delimiter=delimiter, quoting=csv.QUOTE_NONE )
    else : 
        csvfile = open( location ) 
        reader = csv.reader( csvfile, delimiter=delimiter )
    for row in reader :
        if header is None :
            header = row
            continue
        data.append( row )
    return header, data        
    



class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGM():
    """
        知乎-瓦特兰蒂斯
        https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    """
        知乎-瓦特兰蒂斯
        https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class FreeLB():
    """
        1. attack the same as PGD
        2. restore is different
        原始论文: 第一次attack，使用随机初始化的扰动。我们使用初始的梯度值
    """
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad)
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # K次扰动的平均梯度值（current_grad） + 原始梯度（backup_grad）
                param.grad = param.grad + self.grad_backup[name]

def symmetrized_KL_divergence(input, target, alpha=1.0, reduction='batchmean'):
    """
        论文Smart KL-Calc
        https://github.com/namisan/mt-dnn/blob/master/mt_dnn/loss.py#L115
    """
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
        F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
    loss = loss * alpha
    return loss

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    """
        论文Smart KL-Calc
        https://github.com/namisan/mt-dnn/blob/471f717a25ab744e710591274c3ec098f5f4d0ad/mt_dnn/perturbation.py#L91
    """
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    if reduce:
        return (p* (rp- ry) * 2).sum() / bs
    else:
        return (p* (rp- ry) * 2).sum()