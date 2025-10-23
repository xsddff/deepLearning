import re
import collections
def read_dataset(file):
  with open(file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    return [ re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines ]
  
# print(f'文本总行数:{len(lines)}')
# print(lines[0])
# print(lines[10])

def tokenize(lines,token='word'):
  if token == 'word':
    return [ line.split() for line in lines ]
  elif token == 'char':
    return [ list(line) for line in lines]
  else:
    raise ValueError('错误token类型')

# for i in range(10):
#   print(tokens[i])

class Vocab:
  def __init__(self,tokens,min_freq=-1,reserve_tokens=None):
    if tokens is None:
      tokens = []
    if reserve_tokens is None:
      reserve_tokens = []

    counter = collections.Counter( tokens )
    self._token_freqs = sorted(counter.items() , key = lambda x:x[1],reverse=True)
    self.id_to_token = ['<unk>'] + reserve_tokens
    self.token_to_id = {
      token:id for id,token in enumerate(self.id_to_token) 
    }

    for token,freq in self._token_freqs:
      if freq < min_freq:
        break
      self.token_to_id[token] = len(self.id_to_token)
      self.id_to_token.append(token)
  
  def __len__(self):
    return len(self.id_to_token)
  
  def __getitem__(self,tokens):
    if isinstance(tokens,list):
      return [self.__getitem__(token) for token in tokens]
    return self.token_to_id.get(tokens,'0')

  def __call__(self,ids):
    if isinstance(ids,list):
      return [self.__call__(id) for id in ids]
    return self.id_to_token[ids]

  @property
  def token_freqs(self):
    return self._token_freqs    


if __name__ == '__main__':
    lines = read_dataset('RecurrenceNetwork/dataset/time_machine.txt')
    tokens = tokenize(lines)
    corpus = [ tk for token in tokens for tk in token ]
    print(corpus)
    vocab = Vocab(corpus,min_freq=5)
    print(list(vocab.id_to_token)[0:10])
    print(vocab.token_freqs[0:10])