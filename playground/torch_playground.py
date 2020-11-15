import torch

x = torch.tensor([1, 2, 3, 4])
print(x.shape)
x1 = torch.unsqueeze(x, 0)
print(x1)
print(x1.shape)

x2 = torch.unsqueeze(x, 1)
print(x2)
print(x2.shape)

x3 = torch.unsqueeze(x, 2)
print(x3)
print(x3.shape)
