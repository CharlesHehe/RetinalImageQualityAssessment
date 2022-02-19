from torch import tensor, optim, cat

x = tensor([[1, 2, 3], [4, 5, 6]])
y = x[:, 0]
print(x.size())
print(x.stride())
print(y.size())
print(y.stride())
print(x, y)

t = tensor([[1, 2]])
print(t)
x[0, 0] = 9
print(t)
print(x)

# x[0, 0] = 9
#
# print(x.size())
# print(x.stride())
# print(y.size())
# print(y.stride())
# print(x, y)
