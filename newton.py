import numpy as np

a = np.array([[2,4,-2], [1,3,1], [3,2,1]])
b= np.array([14,11,11])
x = np.linalg.solve(a,b)
print(x)

det = np.linalg.det(a)
print(det)