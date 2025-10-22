import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from seq import seqDataLoader
from voc import Vocab,read_dataset,tokenize
from model import RnnModelScratch

def train():

  # 加载数据集 和 词表
  lines = read_dataset('RecurrenceNetwork/dataset/time_machine.txt')
  tokens = tokenize(lines)
  corpus = [ tk for token in tokens for tk in token ]
  voc = Vocab(corpus)

  dataLoader = seqDataLoader(corpus,batch_size=16,num_steps=5,is_random=True)

  # 加载模型
  net = RnnModelScratch(voc_size=len(voc),num_hiddens=256)

  loss = nn.CrossEntropyLoss()
  if isinstance(net,nn.Module):
    opt = torch.optim.SGD(net.parameters(),lr=1)
  else:
    pass
  epoces = 10
  for epoch in range(epoces):
    pass
    



def train_epoch(model,state,train_iter,loss,opt,lr,is_random=False,device='cpu'):
  losses = []
  for X,Y in train_iter:
    if state is None or is_random:
      state = model.begin_state(X.shape[0],device='cpu')
    else:
      for s in state:   # this is for every batch_size H
        s.detach_()    # 把循环神经网络前面的分离，我们只需要它的数值
  
    X.to(device)
    y_pre,state = model(X,state)
    y = F.one_hot(Y.T,num_classes=len(y_pre.shape[-1])).to(device)
    l = loss(y_pre,y).mean()
    opt.zero_grad()
    l.backward()
    clip_grad(model,1)
    opt.forward()
    losses.append( l.item() )
  return losses


def clip_grad(model,theta):
  if isinstance(model,nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]   # model.parameters()  返回的是参数迭代对象，每个迭代对象是各层网络的参数矩阵
  else:
    params: list = model.params

  tmp = sum(torch.sum(p ** 2) for p in params)


  norm = torch.sqrt(tmp)

  if norm > theta:
    for p in params:
      p.grad = p.grad * theta / norm




if __name__ == '__main__':
  train()