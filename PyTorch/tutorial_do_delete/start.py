from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

def initial():
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    print(x.size(), y.size(), z.size())
    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")          # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)                       # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


def autograd():
    x=torch.ones(2,2,requires_grad=True)
    print(x)
    y=x+2
    print(y)
    print(y.grad_fn)
    z=y*y*3
    out=z.mean()
    print(z, out)
    a=torch.randn(2,2)
    a= ((a*3)/(a-1))
    print(a.requires_grad_(True))
    print(a.requires_grad)
    b=(a*a).sum()
    print(b.grad_fn)
    out.backward()
    print(x.grad)
    x=torch.randn(3, requires_grad=True)
    y= x * 2
    while y.data.norm() < 1000:
        y= y * 2
    print(y)
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)
    print(x.grad)
    print(x.requires_grad)
    y=x.detach()
    print(y.requires_grad)
    print(x.eq(y).all())



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



if __name__ == "__main__":
    autograd()
    net = Net()
    print(net)

