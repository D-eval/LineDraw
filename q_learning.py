from linedraw import LineDraw

from preprocessor import FeatureExtractor

import librosa
import soundfile as sf
import sys
sys.path.append("../music-detr")
sys.path.append("../dataset")
from spec.cqt import MultiWindowCQT, get_freqs
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import Counter
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pretty_midi

import random
import torch.nn.functional as F


class Node:
    """
        每个点选择之后的某个点
    """
    def __init__(self, previous_nodes, post_nodes, t, f):
        """
            post_nodes: List int, idx of freq
        """
        self.meta = (t, f)
        
        m = len(previous_nodes)
        n = len(post_nodes)
        
        # state: m + 1, 被任何人连了或者没被连, 频谱关注local_state即可
        # action: n + 1, 去连后人或者谁都不连

        self.state_num = m + 1
        self.action_num = n + 1
        
        self.q_table = np.zeros((self.state_num, self.action_num))
        self.previous_nodes = previous_nodes + [-1] # 被之前某个点连接，或者没人要
        self.next_nodes = post_nodes + [-1] # 和之后的点连接，或者 idle
        
        self.actions = torch.arange(self.action_num).tolist()
        
    def sample(self, state):
        """
        state: 0~m
        post_nodes
        """
        policy = self.q_table[state, :] / (self.q_table[state, :].sum()) # 暂时这样，之后用 mcts
        choice = random.choice(self.actions, self.q_table)
        return choice


class QlineDraw:
    radius = 2
    
    alpha = 0.01 # lr
    gamma = 0.1 # TD decay
    
    event_punishment = 3
    
    def __init__(self, T, F):
        node_lst_tf = []
        for t in range(T):
            node_lst_f = []
            for f in range(F):
                if t==T-1:
                    candidate = torch.arange(max(f-self.radius, 0), min(f+self.radius, F)).tolist()
                    node_lst_f.append(Node(candidate, [], t, f))
                elif t==0:
                    candidate = torch.arange(max(f-self.radius, 0), min(f+self.radius, F)).tolist()
                    node = Node([], candidate, t, f)
                    node_lst_f.append(node)
                else:
                    candidate = torch.arange(max(f-self.radius, 0), min(f+self.radius, F)).tolist()
                    node = Node(candidate, candidate, t, f)
                    node_lst_f.append(node)
            node_lst_tf.append(node_lst_f)
        self.node_lst_tf = node_lst_tf

        self.shape = (T,F)
        
        self.extractor = FeatureExtractor()
        self.linedraw = LineDraw()
        
    def prior(self, x):
        assert x.shape==self.shape
        events = self.linedraw(x)
        # TODO:用 events 初始化 q_table

    def optim_old(self, events, il, t, f, x, num_step=10, lr=1e-3):
        a = torch.tensor(1, requires_grad=True)
        w = torch.tensor(1, requires_grad=True)
        optimizer = torch.optim.Adam([a, w], lr=lr)
        events[il].append((t,f,a,w))
        for step in range(num_step):
            wav = self.linedraw.events_to_wav(events)
            # TODO:这里 cqt 不能 padding
            x_rec, _, _, _ = self.extractor(wav)
            T1 = x_rec.shape[0]
            l1 = ((x[:T1, :] - x_rec)**2).sum()
            
            l1.backward()
                        
            print(a.grad)
            print(w.grad)
            assert 0
            
            optimizer.step()
            events[il][-1] = (t,f,a,w)
            optimizer.zero_grad()
            
        events[il][-1] = (
            t,
            f,
            a.detach().item(),
            w.detach().item()
        )
        return l1.detach().item()
        

    def optim_new(self, events, t, f, x, num_step=10, lr=1e-3):
        a = torch.tensor(1, requires_grad=True)
        w = torch.tensor(1, requires_grad=True)
        optimizer = torch.optim.Adam([a, w], lr=lr)
        events.append([(t,f,a,w)])
        for step in range(num_step):
            wav = self.linedraw.events_to_wav(events)
            # TODO:这里 cqt 不能 padding
            x_rec, _, _, _ = self.extractor(wav)
            T1 = x_rec.shape[0]
            l1 = ((x[:T1, :] - x_rec)**2).sum()
            
            l1.backward()
                        
            print(a.grad)
            print(w.grad)
            assert 0
            
            optimizer.step()
            events[-1] = [(t,f,a,w)]
            optimizer.zero_grad()
            
        events[-1] = [(
            t,
            f,
            a.detach().item(),
            w.detach().item()
        )]
        return l1.detach().item()
        
        
    def episode(self, x):
        alpha = self.alpha
        gamma = self.gamma
        
        assert x.shape==self.shape
        temp_residual_energy = 0
        
        temp_events = [] # [(t,f,a,w)]
        for t in range(self.shape[0]):
            for f in range(self.shape[1]):
                node = self.node_lst_tf[t][f]
                if t==0:
                    state=-1
                else:
                    candidate_previous = node.previous_nodes
                    for line in temp_events:
                        # TODO: 仍然要考虑汇流情况，之后再说
                        state = -1
                        if line[-1][1] in candidate_previous:
                            state = candidate_previous.index(line[-1][1])
                            break
                        
                action = node.sample(state)
                next_node = node.next_nodes[action]
                if next_node == -1:
                    # TODO
                    pass
                    continue
                if t==0:
                    # TODO
                    pass
                    continue
                if t+1 < self.shape[0]:
                    max_future_value = self.node_lst_tf[t+1][next_node].q_table.max()
                else:
                    max_future_value = 0
       
                # instant reward
                if state != -1:
                    previous_f = node.previous_nodes[state]
                    for il, line in enumerate(temp_events):
                        if (line[-1][0]==t-1) & (line[-1][1]==previous_f):
                            residual = self.optim_old(temp_events, il, t, f, x)
                            # residual 越小越好
                            # reward 就是加入动作后 整体 residual 的减少量
                            reward = temp_residual_energy - residual
                            temp_residual_energy = residual
                            break
                else:
                    residual = self.optim_new(temp_events, t, f, x)
                    # residual 越小越好
                    # reward 就是加入动作后 整体 residual 的减少量
                    reward = temp_residual_energy - residual - self.event_punishment
                    temp_residual_energy = residual
                    
                        
                q_s_a = node.q_table[state, action]
                q_s_a = (1-alpha)*q_s_a + alpha*(reward + gamma * max_future_value)
                node.q_table[state, action] = q_s_a
