import numpy as np

x = np.array([3, 4, 2, 1])
print(np.argpartition(x, 2))
print(x[np.argpartition(x, 2)])