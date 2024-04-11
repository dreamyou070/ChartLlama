import torch

a = torch.tensor([1, 2, 3]).to(torch.float32)
a = a.type(torch.long)
print(a.dtype)