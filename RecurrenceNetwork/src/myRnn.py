from turtle import color
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from seq import seqDataLoader
from voc import Vocab,read_dataset,tokenize
from model import RnnModelScratch
from tqdm import tqdm


def train(epoches,device='cpu'):

  # 加载数据集 和 词表
  lines = read_dataset('../dataset/time_machine.txt')
  tokens = tokenize(lines)
  corpus = [ tk for token in tokens for tk in token ]
  voc = Vocab(corpus)

  corpus = voc[corpus]
  # print(corpus)

  dataLoader = seqDataLoader(corpus,batch_size=16,num_steps=5,is_random=True)

  # 加载模型
  net = RnnModelScratch(voc_size=len(voc),num_hiddens=256)

  loss = nn.CrossEntropyLoss()
  if isinstance(net,nn.Module):
    opt = torch.optim.SGD(net.parameters(),lr=1)
  else:
    opt = torch.optim.SGD(net.params,lr=1)
  state = None

  total_losses = []
  with tqdm(total = epoches,desc='模型训练中') as pbar:
    for epoch in range(epoches):
      losses = train_epoch(net,state,dataLoader,loss,opt,is_random=True,device=device)
      total_losses.append(losses)
      if (epoch+1) % 1 == 0:
        pbar.update(1)
        
  return total_losses



def train_epoch(model,state,train_iter,loss,opt,is_random=False,device='cpu'):
  losses = 0
  total = 0
  for X,Y in train_iter:
    if state is None or is_random:
      state = model.begin_state(X.shape[0],device='cpu')
    else:
      for s in state:   # this is for every batch_size H
        s.detach_()    # 把循环神经网络前面的分离，我们只需要它的数值
  
    X.to(device)
    state.to(device)
    y_pre,state = model(X,state)
    y = F.one_hot(Y.T.reshape(-1),num_classes=y_pre.shape[-1]).float().to(device)
    l = loss(y_pre,y).mean()
    opt.zero_grad()
    l.backward()
    clip_grad(model,1)
    opt.step()
    losses += l * X.size(0)
    total += X.size(0)
  return losses/total


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

  total_losses = train(10,'cuda:0')

  plt.plot(range(len(total_losses)),total_losses,color='blue')
  plt.show()