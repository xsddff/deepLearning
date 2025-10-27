import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class my_nn:
  def __init__(self,voc_size,num_hiddens):
    self.voc_size = voc_size
    self.num_hiddens = num_hiddens
  def get_params(self,vocab_size,num_hiddens,device='cpu'):
    W_xh = torch.tensor(np.random.normal(0,0.01,(vocab_size,num_hiddens))).float()
    W_hh = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_hiddens))).float()
    W_hq = torch.tensor(np.random.normal(0,0.01,(num_hiddens,vocab_size))).float()
    b_h = torch.zeros(num_hiddens,dtype=torch.float32).float()
    b_q = torch.zeros(vocab_size,dtype=torch.float32).float()
    params = [W_xh,W_hh,W_hq,b_h,b_q]
    for param in params:
      param.requires_grad_(True)
      param.data = param.data.to(device)
    return params

  def init_rnn_state(self,batch_size,num_hiddens,device='cpu'):
    return torch.zeros((batch_size,num_hiddens),device=device)
    
  def rnn(self,params,voc_size,num_hiddens,device='cpu'):
    '''
    inputs: (seq_len,batch_size,voc_size)
    '''
    W_xh,W_hh,W_hq,b_h,b_q = params

    def forward(inputs,h):
      outputs = []
      for x in inputs:
        h = torch.tanh(x @ W_xh + h @ W_hh + b_h)
        y = h @ W_hq + b_q  # (,batch_size,voc_size)
        outputs.append(y)
      return torch.cat(outputs,dim=0),h     # torch.cat 专门正对张量列表的合并
    # return np.concatenate(outputs,axis=0)
    return forward   # 嵌套函数要将内层的函数暴露出来

class RnnModelScratch(my_nn):
  def __init__(self,voc_size,num_hiddens, device='cpu'):
    super().__init__(voc_size,num_hiddens)
    self.voc_size = voc_size
    self.params = self.get_params(voc_size,num_hiddens,device)
    self.forward_fn = self.rnn(self.params,voc_size,num_hiddens,device)   # self.forward_fn 等待内参函数传入参数
  
  def __call__(self,X,H):
    '''
    X: (batch_size,seq_len)
    '''
    X = F.one_hot(X.T,num_classes=self.voc_size).float()
    return self.forward_fn(X,H)
  def begin_state(self,batch_size,device='cpu'):
    return self.init_rnn_state(batch_size,self.num_hiddens,device)


if __name__ == '__main__':
  # 加载模型
  rnn = RnnModelScratch(100,num_hiddens=256)


