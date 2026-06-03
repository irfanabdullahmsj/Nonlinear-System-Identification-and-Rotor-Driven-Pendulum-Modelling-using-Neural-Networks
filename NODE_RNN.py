import numpy as np

a = np.array([[1],
              [2],
              [3]])

print(a.shape)
# (3,1)

b = a.squeeze()

print(b.shape)
# (3,)