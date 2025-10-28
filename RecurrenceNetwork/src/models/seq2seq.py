from sre_parse import State
from turtle import forward
import torch.nn as nn
import torch

class Encoder(nn.Module):
  def __init__(self,nums_step,features,hiddens,voc_size,num_layers=1,num_directions=1) -> None:
    super().__init__()
    self.hiddens = hiddens
    self.num_layers = num_layers
    self.num_directions = num_directions
    self.embedding = nn.Embedding(nums_step,features)
    self.rnn = nn.RNN(features+hiddens,hiddens)
  
  def forward(self,x):
    '''
    x: (batch_size,nums_step)
    '''
    x = self.embedding(x)
    x = x.permute(1,0,2)
    output,state = self.rnn(x)
    return output,state

class Decoder(nn.Module):
  def __init__(self, nums_step,features,hiddens,voc_size,num_layers=1,num_directions=1) -> None:
    super().__init__()
    self.embedding = nn.Embedding(nums_step,features)
    self.rnn = nn.RNN(features,hiddens)
    self.dense = nn.Linear(hiddens,voc_size)
  
  def init_state(self,enc_output):
    return enc_output[1]
  def forward(self,x,enc_output):
    x = self.embedding(x)
    x = x.permute(1,0,2)   # (time_step,batch_size,features)
    state = self.init_state(enc_output).repeat(x[0],1,1)  # (1,batch_size,hidden_size)
    context = state[-1]
    context_and_x = torch.cat((context,x),-1)
    output,state = self.rnn(context_and_x,state)  # (time_step,batch_size,hidden_size)
    output = nn.ReLU(self.dense(output).permute(1,0,2))
    return output



  