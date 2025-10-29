from d2l import torch as d2l
from numpy import int32
import torch
import torch.utils
import torch.utils.data

def get_dataset(file,n_samples = None):
  with open(file,'r',encoding='utf-8') as f:
    lines = f.read()
  
  lines = lines.replace('\u202f', ' ').replace('\xa0', ' ').lower()
  modify_char = lambda char,pre_char : ' ' + char if char in set('?!.,') and pre_char != ' ' else char
  new_chars = [ modify_char(char,lines[i-1]) for i,char in enumerate(lines) if i>0 ]
  new_lines = ''.join(new_chars)
  target_words,source_words = [],[]
  for i,line in enumerate(new_lines.split('\n')):
    if n_samples and i>n_samples:
      break
    parts = line.split('\t')
    if len(parts) == 2:
      source_words.append(parts[0].split(' '))
      target_words.append(parts[1].split(' '))
  return source_words,target_words

def wordsToArrays(words,num_steps,voc,padding_token='<pad>',endding_token='<eos>'):
  arr = voc[words]
  arr = torch.tensor([ nums[:num_steps] + [voc[endding_token]] if( len(nums) > num_steps ) 
        else nums + [voc[endding_token]] + [voc[padding_token]] * ( num_steps - len(nums) - 1)
        for nums in arr
      ])
  valid_len = (arr != voc[padding_token]).type(torch.int32).sum(1)
  return arr,valid_len

def mntDataLoader(file,batch_size,num_samples,is_train=True):
  file = '../dataset/fra-eng/fra.txt'
  src,tar = get_dataset(file,num_samples)
  voc_src = d2l.Vocab(src,reserved_tokens=['<pad>','<bos>','<eos>'])
  voc_tar = d2l.Vocab(tar,reserved_tokens=['<pad>','<bos>','<eos>'])
  src_mnt_arr,src_valid_len = wordsToArrays(src,5,voc_src)
  tar_mnt_arr,tar_valid_len = wordsToArrays(tar,5,voc_tar)
  data_arrays = (src_mnt_arr,src_valid_len,tar_mnt_arr,tar_valid_len)
  dataset = torch.utils.data.TensorDataset(*data_arrays)
  data_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=is_train)
  return data_iter,voc_src,voc_tar

if __name__ == '__main__':
  file = '../dataset/fra-eng/fra.txt'
  data_iter,voc_src,voc_tar = mntDataLoader(file,16)
  for src,src_len,tar,tar_len in data_iter:
    print(src,src_len)
    print(tar,tar_len)
    break
