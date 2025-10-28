import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class my_nn:
  def __init__(self,voc_size,num_hiddens,num_layers=1):
    self.voc_size = voc_size
    self.num_hiddens = num_hiddens
    self.num_layers = num_layers
  def get_params(self,vocab_size,num_hiddens,device='cpu'):
    W_xh = []
    W_hh = []
    b_h = []
    W_xh_i = torch.tensor(np.random.normal(0,0.01,(vocab_size,num_hiddens))).float()
    for i in range(self.num_layers-1):
      W_xh_ = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_hiddens))).float()
      W_xh.append(W_xh_)
    for i in range(self.num_layers):
      W_hh_ = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_hiddens))).float()
      W_hh.append(W_hh_)
      b_h_ = torch.zeros(num_hiddens,dtype=torch.float32).float()
      b_h.append(b_h_)
    if W_xh != []:
      W_xh = torch.stack(W_xh,dim=0)
    else:
      W_xh = torch.tensor([])
    W_hh = torch.stack(W_hh,dim=0)
    b_h = torch.stack(b_h,dim=0)
    W_hq = torch.tensor(np.random.normal(0,0.01,(num_hiddens,vocab_size))).float()
    b_q = torch.zeros(vocab_size,dtype=torch.float32).float()
    params = [W_xh,W_hh,W_hq,b_h,b_q,W_xh_i]
    for param in params:
      if isinstance(param,list):
        for p in param:
          p.requires_grad_(True)
          p.data = p.data.to(device)
      else:
        param.requires_grad_(True)
        param.data = param.data.to(device)
    return params

  def init_rnn_state(self,num_layers,batch_size,num_hiddens,device='cpu'):
    return torch.zeros((num_layers,batch_size,num_hiddens),device=device)
    
  def rnn(self,params,voc_size,num_hiddens,device='cpu'):
    '''
    inputs: (seq_len,batch_size,voc_size)
    '''
    W_xh,W_hh,W_hq,b_h,b_q,W_xh_i = params

    def forward(inputs,h):   # h -> (nums_layers*num_directions,num_hiddens)
      outputs = []
      for x in inputs:
        pre_layer_h = x @ W_xh_i + h[0] @ W_hh[0] + b_h[0]
        new_h = [ pre_layer_h ]     # 这里不用h[i]重复复制，否则会导致梯度计算版本不一致
        for i in range(1,self.num_layers):  
          layer_h = pre_layer_h @ W_xh[i-1] + h[i] @ W_hh[i] + b_h[i]
          new_h.append(layer_h)
        h = torch.stack(new_h,dim=0)
        y = h[self.num_layers-1] @ W_hq + b_q  # (,batch_size,voc_size)
        outputs.append(y)
      return torch.cat(outputs,dim=0),h    # torch.cat 专门正对张量列表的合并
    # return np.concatenate(outputs,axis=0)
    return forward   # 嵌套函数要将内层的函数暴露出来

class RnnModelScratch(my_nn):
  def __init__(self,voc_size,num_hiddens, device='cpu',num_layers=1,num_directions=1):
    super().__init__(voc_size,num_hiddens)
    self.voc_size = voc_size
    self.num_layers = num_layers
    self.num_directions = num_directions
    self.params = self.get_params(voc_size,num_hiddens,device)
    self.forward_fn = self.rnn(self.params,voc_size,num_hiddens,device)   # self.forward_fn 等待内参函数传入参数
  
  def __call__(self,X,H):
    '''
    X: (batch_size,seq_len)
    '''
    X = F.one_hot(X.T,num_classes=self.voc_size).float()
    return self.forward_fn(X,H)
  def begin_state(self,batch_size,device='cpu'):
    return self.init_rnn_state(self.num_directions*self.num_layers,batch_size,self.num_hiddens,device)


if __name__ == '__main__':
  # 加载模型
  rnn = RnnModelScratch(100,num_hiddens=256)


