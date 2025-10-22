from voc import Vocab
import matplotlib.pyplot as plt
import numpy as np
import torch
import re

with open('../dataset/time_machine.txt') as f:
    lines = f.readlines()
    tokens = [ re.sub('[^A-Za-z0-9]',' ',line) for line in lines]
print(tokens)