import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class my_gru:
  def __init__(self,voc_size,num_hiddens):
    self.voc_size = voc_size
    self.num_hiddens = num_hiddens
  def get_params(self,vocab_size,num_hiddens,device='cpu'):

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((self.voc_size, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xf, W_hf, bf = three() # 遗忘门
    W_xi, W_hi, bi = three() # 输入门
    W_xc, W_hc, bc = three() # 候选记忆门
    W_xo, W_ho, bo = three() # 输出门

    # 输出层参数
    W_hq = normal((num_hiddens, self.voc_size))
    b_q = torch.zeros(self.voc_size, device=device)
    # 附加梯度
    params = [W_xf, W_hf, bf, W_xi, W_hi, bi, W_xc, W_hc, bc, W_xo, W_ho, bo, W_hq, b_q]

    for param in params:
      param.requires_grad_(True)
      param.data = param.data.to(device)
    return params

  def init_lstm_state(self,batch_size,num_hiddens,device='cpu'):
    return ( torch.zeros((batch_size,num_hiddens),device=device),
          torch.zeros((batch_size,num_hiddens),device=device) )
    
  def lstm(self,params,voc_size,num_hiddens,device='cpu'):
    '''
    inputs: (seq_len,batch_size,voc_size)
    '''
    W_xf, W_hf, bf, W_xi, W_hi, bi, W_xc, W_hc, bc, W_xo, W_ho, bo, W_hq, b_q = params

    def forward(inputs,state):
      outputs = []
      h,c = state
      for x in inputs:
        F = torch.sigmoid( x @ W_xf + h @ W_hf + bf )  # 遗忘门
        I = torch.sigmoid( x @ W_xi + h @ W_hi + bi )  # 输入门
        O = torch.sigmoid( x @ W_xo + h @ W_ho + bo )  # 输出门
        C_ = torch.tanh( x @ W_xc + h @ W_hc + bc )  # 候选记忆门
        c = torch.tanh( F * c + I * C_ )
        h = O * c
        y = h @ W_hq + b_q
        outputs.append(y)
      return torch.cat(outputs,dim=0),(h,c)    # torch.cat 专门正对张量列表的合并
    # return np.concatenate(outputs,axis=0)
    return forward   # 嵌套函数要将内层的函数暴露出来

class lstmModelScratch(my_gru):
  def __init__(self,voc_size,num_hiddens, device='cpu'):
    super().__init__(voc_size,num_hiddens)
    self.voc_size = voc_size
    self.params = self.get_params(voc_size,num_hiddens,device)
    self.forward_fn = self.lstm(self.params,voc_size,num_hiddens,device)   # self.forward_fn 等待内参函数传入参数
  
  def __call__(self,X,H):
    '''
    X: (batch_size,seq_len)
    '''
    X = F.one_hot(X.T,num_classes=self.voc_size).float()
    return self.forward_fn(X,H)
  def begin_state(self,batch_size,device='cpu'):
    return self.init_lstm_state(batch_size,self.num_hiddens,device)


if __name__ == '__main__':
  # 加载模型
  rnn = lstmModelScratch(100,num_hiddens=256)


