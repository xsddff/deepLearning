from voc import Vocab
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import random

def readToken():
    with open('RecurrenceNetwork/dataset/time_machine.txt') as f:
        lines = f.readlines()
        tokens = [ re.sub('[^A-Za-z0-9]',' ',line) for line in lines]
    return tokens


def tokenize(lines,token='word'):
    if token == 'word':
        return [ line.split() for line in lines ]
    elif token == 'char':
        return [ list(line) for line in lines ]
    else:
        raise ValueError('格式错误')

def seq_data_iter_random(corpus: list,batch_size,time_step):
    corpus = corpus[random.randint(0,time_step):]
    seq_nums = (len(corpus)-1) // time_step
    indices = corpus[:seq_nums*time_step:time_step]
    random.shuffle(indices)
    
    get_data = lambda indices : corpus[indices:indices+time_step]
    
    batch_nums = seq_nums // batch_size
    for i in range(0,batch_nums*batch_size,batch_size):
        initial_per_batches = indices[i:i+batch_size]
        X = [ get_data(j) for j in initial_per_batches ]
        Y = [ get_data(j+1) for j in initial_per_batches ]
        
        yield torch.tensor(X),torch.tensor(Y)
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class seqDataLoader:
    def __init__(self,corpus,batch_size,num_steps,is_random):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.corpus = corpus
        if is_random:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)
    

if __name__ == '__main__':
    lines = readToken()
    tokens = tokenize(lines)

    # test for the differ time step tokens freq
    voc1 = Vocab(tokens)
    token2 = [ [tk] for token in tokens for tk in zip(token[:-2],token[1:]) ]
    voc2 = Vocab(token2)
    token3 = [ [tk] for token in tokens for tk in zip(token[:-3],token[1:-2],token[2:]) ]
    voc3 = Vocab(token3)
    # print(voc3.token_freqs)
    plt.plot(range(len(voc1.token_freqs[0:10])),[ freq for tk,freq in voc1.token_freqs[0:10] ],color='blue')
    plt.show()

    plt.plot(range(len(voc2.token_freqs[0:10])),[ freq for tk,freq in voc2.token_freqs[0:10] ],color='blue')
    plt.show()

    plt.plot(range(len(voc3.token_freqs[0:10])),[ freq for tk,freq in voc3.token_freqs[0:10] ],color='blue')
    plt.show()

    corpus : list = [ tk for token in tokens for tk in token ]
    my_seq = list(range(35))
    for X,Y in seq_data_iter_random(my_seq,2,5):
        print('X:',X,'\nY:',Y)
