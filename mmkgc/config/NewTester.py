# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
import pandas as pd
from tqdm import tqdm

class NewTester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True, other_model=None, norm=False, mu=0.5):

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.other_model = other_model
        self.norm = norm
        self.mu = mu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        print(data)
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
        
    def run_link_prediction_old(self, type_constrain=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = self.data_loader
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        return mrr, mr, hit10, hit3, hit1
    
    def run_link_prediction(self, type_constrain=False):
        df = pd.read_csv(self.model)
        l_filter_reci_rank = df.loc[0, 'MRR']
        r_filter_reci_rank = df.loc[1, 'MRR']
        avg_reci_rank       = df.loc[2, 'MRR']

        l_filter_rank = df.loc[0, 'MR']
        r_filter_rank = df.loc[1, 'MR']
        avg_rank      = df.loc[2, 'MR']

        l_filter_tot = df.loc[0, 'hit@10']
        r_filter_tot = df.loc[1, 'hit@10']
        avg_tot      = df.loc[2, 'hit@10']

        l3_filter_tot = df.loc[0, 'hit@3']
        r3_filter_tot = df.loc[1, 'hit@3']
        avg3          = df.loc[2, 'hit@3']

        l1_filter_tot = df.loc[0, 'hit@1']
        r1_filter_tot = df.loc[1, 'hit@1']
        avg1          = df.loc[2, 'hit@1']

        # Print results
        print("metric:\t	MRR\t\tMR\t\thit@10\t hit@3\t hit@1")
        print(f"l(filter):\t{l_filter_reci_rank:.4f}\t{l_filter_rank:.4f}\t{l_filter_tot:.4f}\t{l3_filter_tot:.4f}\t{l1_filter_tot:.4f}")
        print(f"r(filter):\t{r_filter_reci_rank:.4f}\t{r_filter_rank:.4f}\t{r_filter_tot:.4f}\t{r3_filter_tot:.4f}\t{r1_filter_tot:.4f}")
        print(f"averaged(filter):\t{avg_reci_rank:.4f}\t{avg_rank:.4f}\t{avg_tot:.4f}\t{avg3:.4f}\t{avg1:.4f}\n")
        return avg_reci_rank, avg_rank, avg_tot, avg3, avg1

