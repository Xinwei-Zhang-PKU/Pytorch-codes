# Reference: https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=1658s
import torch

t1 = torch.tensor(4.)
print(t1)
print(t1.dtype)
print(t1.shape)
# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
print(x)
print(w)
print(b)
y = w * x + b
print(y)
# Compute derivatives
y.backward()
# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
# Create a tensor with a fixed value for every element
t6 = torch.full((3, 2), 42)
print(t6)
# Matrix
t3 = torch.tensor([[5., 6],
                   [7, 8],
                   [9, 10]])
# Concatenate two tensors with compatible shapes
t7 = torch.cat((t3, t6))
print(t7)
