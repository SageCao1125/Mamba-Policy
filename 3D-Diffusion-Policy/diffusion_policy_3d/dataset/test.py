import torch
import numpy as np

from collections import defaultdict, deque

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    else:
        return np.array(x[-n:])

# a = torch.rand(10,24)
a = np.random.rand(10,24)

print(a.shape, a[...,:].shape)

b = deque(maxlen=3)
b.append(1)
print(b)
b.append(2)
print(b)
b.append(3)
print(b)
b.append(4)
print(b)
b.append(5)
print(b)

print(np.array(b), np.array(b).shape)

c = take_last_n(b, 2)
print(c, c.shape)

