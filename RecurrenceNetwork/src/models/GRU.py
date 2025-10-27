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

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, self.voc_size))
    b_q = torch.zeros(self.voc_size, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
      param.requires_grad_(True)
      param.data = param.data.to(device)
    return params

  def init_gru_state(self,batch_size,num_hiddens,device='cpu'):
    return torch.zeros((batch_size,num_hiddens),device=device)
    
  def gru(self,params,voc_size,num_hiddens,device='cpu'):
    '''
    inputs: (seq_len,batch_size,voc_size)
    '''
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params

    def forward(inputs,h):
      outputs = []
      for x in inputs:
        R = torch.sigmoid(x @ W_xr + h @ W_hr + b_r)  # 重置门
        Z = torch.sigmoid(x @ W_xz + h @ W_hz + b_z)  # 更新门
        h_ = torch.tanh( x @ W_xh + (R * h) @ W_hh+ b_h ) # 候选隐状态
        h = Z * h + ( 1-Z ) * h_
        y = h @ W_hq + b_q
        outputs.append(y)
      return torch.cat(outputs,dim=0),h     # torch.cat 专门正对张量列表的合并
    # return np.concatenate(outputs,axis=0)
    return forward   # 嵌套函数要将内层的函数暴露出来

class GRUModelScratch(my_gru):
  def __init__(self,voc_size,num_hiddens, device='cpu'):
    super().__init__(voc_size,num_hiddens)
    self.voc_size = voc_size
    self.params = self.get_params(voc_size,num_hiddens,device)
    self.forward_fn = self.gru(self.params,voc_size,num_hiddens,device)   # self.forward_fn 等待内参函数传入参数
  
  def __call__(self,X,H):
    '''
    X: (batch_size,seq_len)
    '''
    X = F.one_hot(X.T,num_classes=self.voc_size).float()
    return self.forward_fn(X,H)
  def begin_state(self,batch_size,device='cpu'):
    return self.init_gru_state(batch_size,self.num_hiddens,device)


if __name__ == '__main__':
  # 加载模型
  rnn = GRUModelScratch(100,num_hiddens=256)


